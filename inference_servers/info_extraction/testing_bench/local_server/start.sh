#!/bin/bash
#downloading of GAIA model from ollama repository.

sudo ollama serve
ollama -v && sudo ollama pull cnmoro/gemma3-gaia-ptbr-4b:q8_0