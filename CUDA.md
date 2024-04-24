# CUDA
CUDA: GPU Programing

## Alapok
C++ nyelvet használ.

Ezek kellenek, hogy tudjunk GPU-re programot írni.
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
```

## konstans értékek

Lehet konstans értékeket beállítani, amikre később lehet hivatkozni:
```
#define N 5000
#define BLOCK_SIZE 500
#define BLOCK 10
```

## Fg és tárolók elhelyezkedése, létrehozása

Ezzel lehet a CPU-ra tömböt létrehozni: (CSAK A CPU-n létezik!)
```
int A[] = { 1,2,3,4,5 };
```

Ezzel lehet a GPU-n létrehozni változót, ami a GPU memória területére jön létre. 
```
__device__ int dev_A[5];
```
```
__device__ float *devPtr;
__device__ float devPtr[1024];
```

Ezzel lehet a GPU-n létrehozni konstanst:
```
__constant__ float *devPtr;
__constant__ float devPtr[1024];
```

Ezzel az előtaggal lehet olyan függvényt létrehozni, amit CPU-n és GPU-n is lehet futtatni.
```
__global__ void fg() { }
```

A GPU blokkon belül vannak még memória, ahova át lehet helyezni adatokat, de csak Globális fg-ben lehet rájuk hivatkozni:
```
_shared__ int shr_A[5];
```

Felszabadítás:
```
float *dev_Ptr;
cudaMalloc((void**)&dev_Ptr, 256 * sizeof(float));
cudaFree(dev_Ptr);
```

## GPU indexek

Az adott blokkban a száll index lekérése: (Ez csak a blokkos indexet adja meg majd, más szóval ebből lehet több is, mivel több blokkot is futatthatunk majd)
```
int i = threadIdx.x;
```

Az adott blokk indexét adja vissza amiben van:
```
int i = blockIdx.x;
```

Az adott blokk hosszát adja vissza:
```
int i = blockDim.x;
```

## Szállak bevárása

Meg lehet oldani, hogy a szállak bevárják egymást és ami után minden száll ugyan ott van, csak akkor lépjenek tovább.
```
__syncthreads();
```
## Main metódus

CPU-ról GPU-ra másolni az adatok így lehet:
```
cudaMemcpyToSymbol(dev_A, &A, 5*sizeof(int)); 
```
vagy 
```
int* intInGPU = nullptr;
cudaMalloc(&originalImgInGPU, sizeof(int));
cudaMemcpy(intInGPU, intValahonnan, sizeof(int), cudaMemcpyHostToDevice);
```
1. dev_A a GPU-s változót tartalmazza (memóriában a helyét)
2. &A a CPU-n az adatot, amit másolunk (lehetne így is: &(A[0]), A)
3. 5*sizeof(int) mekkora az átmásolt adat mérete

A GPU-n fg futtatása:
```
fg <<<1, 5 >>> ();
```
Ahol az 1-es most a blokkot jelenti, hogy 1 blokkot indítunk és az 5-ös pedig azt, hogy mennyi szállat abban.

GPU-ról CPU-ra vissza másolás:
```
cudaMemcpyFromSymbol(A, dev_A, 5 * sizeof(int));
```
vagy
```
cudaMemcpy(intValahonnan, intInGPU, sizeof(int), cudaMemcpyDeviceToHost);
```
1. A a CPU-n az adatot, amit másolunk
2. dev_A a GPU-s változót tartalmazza
3. 5*sizeof(int) mekkora az átmásolt adat mérete

Console-ra kiírás:
```
printf("A[%d]=%d\n", i, A[i]);
```

## Segéd képletek

Vissza adja a megfelelő tömb indexet:
```
int Global_x_component = blockIdx.x * blockDim.x + threadIdx.x
```

## GPU-n belüli szorzás kód
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int A[] = { 1,2,3,4,5 };
__device__ int dev_A[5];

int b = 5;
__device__ int dev_b;

__global__ void szorzas(int mennyivel) {
	int i = threadIdx.x;
	dev_A[i] *= mennyivel;
}

int main() {
	cudaMemcpyToSymbol(dev_b, &b, sizeof(int));
	cudaMemcpyToSymbol(dev_A, &(A[0]), 5* sizeof(int));
	szorzas <<<1, 5 >>> (3);
	cudaMemcpyFromSymbol(A, dev_A, 5 * sizeof(int));
	cudaMemcpyFromSymbol(&b, dev_b, sizeof(int));
	for (int i = 0; i < 5; i++)
	{
		printf("A[%d]=%d\n", i, A[i]);
	}
}
```
## Szövegben szó keresés
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

std::string szoveg = "ababcab";
std::string szo = "abc";
const int n = (int)szoveg.size();
int m = (int)szo.size();
char szovegTomb[];
__device__ std::string dev_szoveg;
__device__ std::string dev_szo;

int hol = -1;
__device__ int dev_hol;

void keresesCPU()
{
	for (int i = 0; i <= n-m; ++i)
	{
		if (szoveg[i] == szo[0] && szoveg[i + 1] == szo[1] & szoveg[i + 2] == szo[2])
		{
			hol = i;
		}
	}
}

void keresesGPU()
{
	int i = threadIdx.x;
	if (dev_szoveg[i] == dev_szo[0] && dev_szoveg[i + 1] == dev_szo[1] & dev_szoveg[i + 2] == dev_szo[2])
	{
		dev_hol = i;
	}
}

int main() {

	//keresesCPU();

	cudaMemcpyToSymbol(&dev_szoveg, &szoveg, n);
	cudaMemcpyToSymbol(&dev_szo, &szo, m);
	cudaMemcpyToSymbol(&dev_hol, &hol, sizeof(int));
	keresesGPU <<<1, n - m+1 >>> ();
	cudaMemcpyFromSymbol(&szoveg, &dev_szoveg, n);
	cudaMemcpyFromSymbol(&szo, &dev_szo, m);
	cudaMemcpyFromSymbol(&hol, &dev_hol, sizeof(int));
	printf("%d", hol);
}
```

## Futószalag rendezés + szimpla szorzás

```

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 5000
#define BLOCK_SIZE 500
#define BLOCK 10

int A[N];

__device__ int dev_A[N];

__global__ void Multiply() //simple multiply
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dev_A[i] *= 2;
}

__global__ void OneBlockSort()
{
    for (int i = 1; i <= N/2; i++)
    {
        for (int shift = 0; shift <= 1; shift++)
        {
            int chk = 2 * threadIdx.x - shift;
            if ((threadIdx.x > 0 || shift != 1) && dev_A[chk] > dev_A[chk + 1])
            {
                int temp = dev_A[chk];
                dev_A[chk] = dev_A[chk + 1];
                dev_A[chk + 1] = temp;
            }
            __syncthreads();
        }
    }
}

__global__ void MultipleBlockSort()
{
    for (int i = 1; i <= blockDim.x; i++)
    {
        for (int shift = 0; shift <= 1; shift++)
        {
            int chk = 2 * (blockIdx.x * blockDim.x + threadIdx.x) - shift;
            if ((threadIdx.x > 0 || shift != 1) && dev_A[chk] > dev_A[chk + 1])
            {
                int temp = dev_A[chk];
                dev_A[chk] = dev_A[chk + 1];
                dev_A[chk + 1] = temp;
            }
            __syncthreads();
        }
    }
}

__global__ void MultipleBlockSortWithShare()
{
    __shared__ int shr_A[BLOCK_SIZE * 2];
    shr_A[2 * threadIdx.x] = dev_A[(blockIdx.x * blockDim.x + threadIdx.x) * 2];
    shr_A[2 * threadIdx.x + 1] = dev_A[(blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1];
    __syncthreads();

    for (int i = 1; i <= blockDim.x; i++)
    {
        for (int shift = 0; shift <= 1; shift++)
        {
            int chk = 2 * threadIdx.x - shift;
            if ((threadIdx.x > 0 || shift != 1) && shr_A[chk] > shr_A[chk + 1])
            {
                int temp = shr_A[chk];
                shr_A[chk] = shr_A[chk + 1];
                shr_A[chk + 1] = temp;
            }
            __syncthreads();
        }
    }

    dev_A[(blockIdx.x * blockDim.x + threadIdx.x) * 2] = shr_A[2 * threadIdx.x];
    dev_A[(blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1] = shr_A[2 * threadIdx.x + 1];
}

int main()
{
    int tolt = N;
    for (int i = 0; i < N; i++)
    {
        A[i] = tolt;
        tolt--;
    }

    cudaMemcpyToSymbol(dev_A, A, N * sizeof(int));
    MultipleBlockSortWithShare <<< BLOCK, BLOCK_SIZE >>> ();
    cudaMemcpyFromSymbol(A, dev_A, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        printf("A[%d] = %d\n", i, A[i]);
    }

    return 0;
}
```

Optimalizált mátrix művelet kódja:
```
__global__ void MatrixMulGPUTiled(float *devA, float *devB, float *devC) {
	__shared__ float shr_A[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float shr_B[BLOCK_SIZE][BLOCK_SIZE];
	int indx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int indy = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	float c = 0;
	for (int k = 0; k < N / BLOCK_SIZE; k++) {
		shr_A[threadIdx.y][threadIdx.x] = devA[k * BLOCK_SIZE + threadIdx.x + indy * N];
		shr_B[threadIdx.y][threadIdx.x] = devB[indx + (k* BLOCK_SIZE + threadIdx.y) * N];
		__syncthreads();
		for (int l = 0; l < BLOCK_SIZE; l++) {
			c += shr_A[threadIdx.y][l] * shr_B[l][threadIdx.x];
		}
		__syncthreads();
	}
	devC[indx + indy * N] = c;
}
```

## Atomi műveletek
### Összeadás
Összeadja a régit és az újat, majd vissza adja a régi adatot.
```
int atomicAdd(int* address, int val)
```

### Kivonás
Kivonja a régiből az újat, majd vissza adja a régi adatot.
```
int atomicSub(int* address, int val)
```

### Növelés
2. paramétert ha túl lépi, akkor 0 lesz az eredménye és azt adja vissza ami lett.
```
unsigned int atomicInc(unsigned int* address, unsigned int val)
```

### Csökkentés
2. paramétert ha túl lépi, akkor az lesz az eredménye és azt adja vissza.
```
unsigned int atomicInc(unsigned int* address, unsigned int val)
```

### Min
Ha az address nagyobb mint a kapott érték, akkor az addresnek az lesz az új értéke és a régi adatot adja vissza.
```
int atomicMin(int* address, int val)
```

### Max
Ugyan az mint min, csak maxba
```
int atomicMax(int* address, int val)
```

### Csere
Az address-nek a val lesz az új értéke és vissza adja a régit.
```
int atomicExch(int* address, int val)
```

### Megvizsgálás és csere
Address az hasonló a compare-rel, akkor megváltozik a val-ra és vissza adja a régi értéket.
```
int atomicCAS(int* address, int compare, int val)
```

### Példa az atomicra, minimum kiválasztás
```
__global__ static void MinSearch(int *devA) {
	__shared__ int localMin[BlockN*2];
	int blockSize = BlockN;
	int itemc1 = threadIdx.x * 2;
	int itemc2 = threadIdx.x * 2 + 1;
	for(int k = 0; k <= 1; k++) {
		int blockStart = blockIdx.x * blockDim.x * 4 + k * blockDim.x * 2;
		int loadIndx = threadIdx.x + blockDim.x * k;
		if (blockStart + itemc2 < N) {
			int value1 = devA[blockStart + itemc1];
			int value2 = devA[blockStart + itemc2];
			localMin[loadIndx] = value1 < value2 ? value1 : value2;
		} else
			if (blockStart + itemc1 < N)
				localMin[loadIndx] = devA[blockStart + itemc1];
			else
				localMin[loadIndx] = devA[0];
	}
	__syncthreads();
	while (blockSize > 0) {
		int locMin = localMin[itemc1] < localMin[itemc2] ? localMin[itemc1] : localMin[itemc2];
		__syncthreads();
		localMin[threadIdx.x] = locMin;
		__syncthreads();
		blockSize = blockSize / 2;
	}
	if (threadIdx.x == 0) atomicMin(devA, localMin[0]);
}
```

## Optimalizálás


### Mennyi GPU-unk van?
```
int deviceCount;
cudaGetDeviceCount(&deviceCount);
```

### Hogyan választsuk ki a videó kártyát?
```
int deviceNumber = 0;
cudaSetDevice(deviceNumber);
```

### GPU adatok lekérdezése
```
int deviceNumber = 1;
cudaDeviceProperty deviceProp;
cudaGetDeviceProperties(&deviceProp, deviceNumber);
```
#### cudaDeviceProperty lekérdezhető adatok
- name: Name of the device
- totalGlobalMem: Size of the global memory
- sharedMemPerBlock: Size of the shared memory per block
- regsPerBlock: Number of registers per block
- totalConstMem: Size of the constant memory
- warpSize: Size of the warps (32)
- maxThreadsPerBlock: Maximum number of threads by block
- maxThreadsDim: Maximum dimension of thread blocks
- maxGridSize: Maximum grid size
- clockRate: Clock frequency
- minor, major: Version numbers
- multiprocessorCount: Number of multiprocessors
- deviceOverlap: Is the device capable to overlapped read/write

### Kihasználhatatóság
Occupancy = Active Warps / Maximum Number of Warps. De ez mástól is függ:
Occupancy is limited by:
- Max Warps or Max Blocks per Multiprocessor
- Registers per Multiprocessor
- Shared memory per Multiprocessor
Occupancy = Min( register occ., shared mem occ., block size occ.)

```
cudaSetDevice(0);
float A[N][N], B[N][N], C[N][N]; float *devA, *devB, *devC;
cudaMalloc((void**) &devA, sizeof(float) * N * N);
cudaMalloc((void**) &devB, sizeof(float) * N * N);
cudaMalloc((void**) &devC, sizeof(float) * N * N);
cudaMemcpy(devA, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
cudaMemcpy(devB, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
dim3 grid((N - 1) / BlockN + 1, (N - 1) / BlockN + 1);
dim3 block(BlockN, BlockN);
MatrixMul<<<grid, block>>>(devA, devB, devC);
cudaThreadSynchronize();
cudaMemcpy(C, devC, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
cudaFree(devA); cudaFree(devB); cudaFree(devC);
```
