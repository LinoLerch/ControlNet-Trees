
from transformers import ImageToTextPipeline
from transformers.image_utils import load_image

class CaptionPipeline(ImageToTextPipeline):
    """
    Customized ImageToTextPipeline to also return filename allowing to match the generated caption to the right image.
    
    Based on original ImageToTextPipeline code: https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/pipelines/image_to_text.py#L108C5-L108C46
    """
    def preprocess(self, image, prompt=None):
        filename = image.filename.split("\\")[-1]
        image = load_image(image)
        model_inputs = self.image_processor(images=image, return_tensors=self.framework)
        if self.model.config.model_type == "git" and prompt is None:
            model_inputs["input_ids"] = None
        
        return filename, model_inputs
    
    def _forward(self, model_inputs, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}
        
        filename, model_inputs = model_inputs
        
        inputs = model_inputs.pop(self.model.main_input_name)
        model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
        return filename, model_outputs

    def postprocess(self, model_outputs):
        records = []
        filename, model_outputs = model_outputs
        for output_ids in model_outputs:
            record = {
                "filename": filename,
                "generated_text": self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
            }
            records.append(record)
        return records