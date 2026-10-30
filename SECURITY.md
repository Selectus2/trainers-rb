# Security Policy

## Supported versions

Only the latest released version of trainers-rb receives security fixes.

## Reporting a vulnerability

Please do not open a public issue. Email **vishwajeetsinghd@gmail.com** with:

- a description of the problem and its impact
- the trainers-rb, torch-rb and Ruby versions affected
- steps or a script to reproduce it

You will get a reply within 7 days. Please give us a reasonable window to release a fix before disclosing publicly.

## Scope

trainers-rb loads and saves model weights. Weight files are read with safetensors, which does not execute code. Treat any model, tokenizer or adapter file from an untrusted source with the same care you would give untrusted input. Problems in torch-rb, LibTorch, transformers-rb or safetensors should be reported to those projects.
