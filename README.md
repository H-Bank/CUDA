# CUDA
CUDA: GPU Programing

## Alapok

Ezek kellenek, hogy tudjunk GPU-re programot írni.
- #include "cuda_runtime.h"
- #include "device_launch_parameters.h"
- #include <stdio.h>

## Tárolók

Ezzel lehet a CPU-ra tömböt létrehozni: (CSAK A CPU-n létezik!)
- int A[] = { 1,2,3,4,5 };
