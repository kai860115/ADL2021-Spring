from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data.dataset import Dataset

from transformers import Trainer
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.debug_utils import DebugOption

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

class SummarizationTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, gen_config=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.gen_config = gen_config
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        compute_metrics = self.compute_metrics 
        self.compute_metrics = None
        try:
            eval_loop = self.prediction_loop# if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(self.eval_examples, output, 'eval')
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        self.log(metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def predict(
        self,
        test_dataset: Dataset,
        test_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)

        compute_metrics = self.compute_metrics 
        self.compute_metrics = None
        try:
            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        eval_preds = self.post_process_function(test_examples, output, 'eval')
        metrics = self.compute_metrics(eval_preds) if 'title' in test_examples.features else None

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            **self.gen_config,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)
        
        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor