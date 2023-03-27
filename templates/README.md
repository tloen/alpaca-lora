# Prompt templates for llama/Alpaca

Several templates styles can be used for tuning and querying the model.

This directory contains some common ones to be used with existing LoRAs or to train your owns.

## Format

Each template is a json, extended from the legacy Alpaca directory.

Original keys:  
- prompt_input : The template to use when input is not None. Uses {instruction} and {input} placeholders.
- prompt_no_input : The template to use when input is None. Uses {instruction} placeholders.

The following keys were added:  
- description : A short description of the template, with possible use cases.
- response_split : The text to use as separator when cutting real response from the model output.

No {response} placeholder was used, since the response is always the last element of the template and is just to be concatenated to the rest.

## Example template

The default template, used unless otherwise specified, is alpaca.json

```
```

## Current templates

### alpaca

Default template used for generic LoRA fine tunes so far.

### alpaca_legacy

Legacy template used by the original alpaca repo, with no `\n` after the response field. Kept for reference and experiments.

### alpaca_short

A trimmed down alpaca template, that seem to perform just as well and spare some tokens. Models created with the default template seem to be query-able by the short tempalte as well. More experiments are welcome.

### vigogne

The default alpaca template, translated to french. This template was used to train the "Vigogne" LoRA and is to be used to query it, or for extra fine tuning.
