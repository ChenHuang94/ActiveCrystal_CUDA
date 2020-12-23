/**
 * @author {Chen Huang}
 * @email {hcyouxiangd@163.com}
 *
 * This CUDA code is used to generate configurations of Active Crystal systems, 
 * where the active particles form a triangular crystal and are connected permanently by springs.
 * To run this file, remember to create a new folder named "dataFiles" in the same folder.
 * Then use the command: "nvcc -std=c++11 kernal.cu -o test" to generate an excetable file.
 */
#include <stdio.h>
#include <iostream>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>
#define _USE_MATH_DEFINES // use M_PI
#include <math.h>
#include <random>
#include <fstream> // ofstream header file
#include <sstream> // string stream
#include <curand.h> // cuRAND library
#include <time.h> // timer
using namespace std;
#define LINEARSIZE 128// number of particles along x axis
#define NUM_PARTICLES LINEARSIZE*LINEARSIZE
#define SHARED_BUFFER_SIZE (NUM_PARTICLES>1024? 1024:32) //32, can not exceed 1024!
#define A0 1.0f // lattice constant a0
#define K 100.0f // spring constant
#define B 0.0f // initial value of active force
#define C 0.0f // rate of change of direction, align to force // 5.0f 
#define D 0.0f // rate of change of direction, align to mean neighbor velocity (Vicsek model) //0.2
#define D_T 0.01f // noise D_T
#define D_theta 0.3f // noise D_theta
#define Delta_t 0.001f // Integration time
#define SQRT_2_Delta_t_D_T sqrtf(2.0f*Delta_t*D_T) // translational noise 
#define SQRT_2_Delta_t_D_theta sqrtf(2.0f*Delta_t*D_theta) // angular noise

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef struct {
	float x, y, theta;//, strain;
} particle_t;

// Normalize particle index considering PBC.
int normal(int i, int n)
{
	if (i < 0)			i += n;
	else if (i > n - 1)	i -= n;
	return i;
}

// Normalize particle direction into interval (0,2 pi).
__host__ __device__ float normal_direction(float angle)
{
	float pi = (float)M_PI;
	float two_pi = 2.0f*(float)M_PI;
	while (angle < -pi)
		angle += two_pi;
	while (angle > pi)
		angle -= two_pi;
	return angle;
}

// Find 6 neighbors for each particle, return (NeighbourList) of size (1,LINEARSIZE*LINEARSIZE).
void find6neighbors(int* NeighbourList)
{
	for (int i = 0; i < LINEARSIZE; i++) { // column index
		for (int j = 0; j < LINEARSIZE; j++) { // row index
			NeighbourList[(j*LINEARSIZE + i) * 6 + 0] = normal(j - 1, LINEARSIZE)*LINEARSIZE + normal(i, LINEARSIZE);
			NeighbourList[(j*LINEARSIZE + i) * 6 + 1] = normal(j - 1, LINEARSIZE)*LINEARSIZE + normal(i + 1, LINEARSIZE);
			NeighbourList[(j*LINEARSIZE + i) * 6 + 2] = normal(j, LINEARSIZE)*LINEARSIZE + normal(i + 1, LINEARSIZE);
			NeighbourList[(j*LINEARSIZE + i) * 6 + 3] = normal(j + 1, LINEARSIZE)*LINEARSIZE + normal(i, LINEARSIZE);
			NeighbourList[(j*LINEARSIZE + i) * 6 + 4] = normal(j + 1, LINEARSIZE)*LINEARSIZE + normal(i - 1, LINEARSIZE);
			NeighbourList[(j*LINEARSIZE + i) * 6 + 5] = normal(j, LINEARSIZE)*LINEARSIZE + normal(i - 1, LINEARSIZE);
		}
	}
}

void initTriangleGrid(particle_t* particles, int* NeighbourList)
{
	find6neighbors(NeighbourList);

	//// Generate new positions
	for (int i = 0; i < LINEARSIZE; i++) { // column index
		for (int j = 0; j < LINEARSIZE; j++) { // row index
			particles[j*LINEARSIZE + i].y = A0 * ((sqrtf(3.0f) / 2.0f * (float)j));// +0.1f*cosf(1.0f* particles[j*LINEARSIZE + i].x) + 0.2f*sinf(2.3f* particles[j*LINEARSIZE + i].x + (float)M_PI / 6.0f);
			particles[j*LINEARSIZE + i].theta = (float)rand() / RAND_MAX*2.0f*M_PI;//0.5f*((float)rand() / RAND_MAX - 0.5f)*2.0f+ (float)M_PI;
			//particles[j*LINEARSIZE + i].strain = 1.0f;
			if ((j + 1) % 2 == 1)
				particles[j*LINEARSIZE + i].x = A0 * ((float)j / 2.0f + (float)i);
			else
				particles[j*LINEARSIZE + i].x = A0 * (0.5f + ((float)j - 1.0f) / 2.0f + (float)i);


		}
	}

	////// Read from file
	//vector<double> V;
	//vector<double>::iterator it;
	//ifstream infile("..\\dataFiles_local\\data795.txt");
	//string s; 
	//getline(infile, s); // only read the first line, do nothing
	//float d;
	//for (int i = 0; i < NUM_PARTICLES; i++) {
	//	infile >> particles[i].x;
	//	infile >> particles[i].y;
	//	infile >> particles[i].theta;
	//	infile >> particles[i].strain;
	//}
	//infile.close();
}

// Correct a distance vector accoring to periodic boundary conditions.
__device__ void distPBC2D(float *dx, float *dy)
{
	// transform to tilted coordinate system which is easier to compute,
	// where the angle between x'and y'is 60 degree: x'=x-y/sqrt(3), y'=y/sqrt(3)*2
	float dx_prime = *dx - *dy / sqrtf(3.0f);
	float dy_prime = *dy / sqrtf(3.0f) * 2.0f;
	float L = LINEARSIZE*A0;
	// dx
	if (dx_prime >= L / 2.0f)
		dx_prime = dx_prime - L;
	else if (dx_prime < -L / 2.0f)
		dx_prime = dx_prime + L;
	// dy
	if (dy_prime >= L / 2.0f)
		dy_prime = dy_prime - L;
	else if (dy_prime < -L / 2.0f)
		dy_prime = dy_prime + L;

	// transform(vec_prime) back to the original coordinate system: x=x'+y'/2, y=y'/2*sqrt(3)
	*dx = dx_prime + dy_prime / 2.0f;
	*dy = dy_prime / 2.0f * sqrt(3.0f);
}

// Kernel function: Time integration of the Overdamped Langevin Dynamics
__global__ void Kerneltimestep_Overdamped(particle_t* dev_particles, int* dev_NeighbourList, float* dev_ElasticForce,
	float* dev_vel_Noise, float* dev_direction_Noise, float2* dev_particle0xy)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < NUM_PARTICLES)
	{
		// Get particle to be updated by this thread.
		particle_t particle = dev_particles[id];
		float dx, dy, dr_norm;//, strain;
		float2 f_elast = { 0.0f, 0.0f },
			n_nei = { 0.0f, 0.0f };
		float num_neighbor = 6.0f; // number of neighbors of particle (id)
		for (int j = 0; j < 6; j++) {
			particle_t other = dev_particles[dev_NeighbourList[id * 6 + j]];
			dx = other.x - particle.x;
			dy = other.y - particle.y;
			distPBC2D(&dx, &dy); // % Fix (dx,dy) upon PBC
			dr_norm = sqrtf(dx*dx + dy*dy); // distance between particle (id) and its (j)th neighbor
			f_elast.x += dx / dr_norm*(dr_norm - A0);
			f_elast.y += dy / dr_norm*(dr_norm - A0);
			//strain += fabsf(dr_norm - A0) / A0;

			n_nei.x += cosf(other.theta);
			n_nei.y += sinf(other.theta);
		}
		f_elast.x = K*f_elast.x;
		f_elast.y = K*f_elast.y;

		// normalize (n_nei)
		n_nei.x = n_nei.x / num_neighbor;
		n_nei.y = n_nei.y / num_neighbor;
		//particle.strain = strain / num_neighbor;

		//dev_ElasticForce[id * 2 + 0] = f_elast.x;	dev_ElasticForce[id * 2 + 1] = f_elast.y;

		float2 n_i = { cosf(particle.theta),sinf(particle.theta) }; // direction of active force if particle (i)

		float del_x = (B*n_i.x + f_elast.x)*Delta_t + SQRT_2_Delta_t_D_T*dev_vel_Noise[id * 2 + 0];
		float del_y = (B*n_i.y + f_elast.y)*Delta_t + SQRT_2_Delta_t_D_T*dev_vel_Noise[id * 2 + 1];
		particle.x += del_x;
		particle.y += del_y;

		//// compute (f_total), use (n\times f \cdot \hat{z})
		float2 f_total;
		f_total.x = C*(f_elast.x) + D*n_nei.x;
		f_total.y = C*(f_elast.y) + D*n_nei.y;
		float del_theta = (n_i.x*f_total.y - n_i.y*f_total.x)*Delta_t + SQRT_2_Delta_t_D_theta*dev_direction_Noise[id];
		particle.theta = normal_direction(particle.theta + del_theta); // normalize angle into interval (0,2*pi)
		
		dev_particles[id] = particle; // Write new particle.{x,y,theta} back to global memory.

		if (id == 0) // to output particle0's x and y coordinate
		{
			dev_particle0xy[0].x = particle.x;
			dev_particle0xy[0].y = particle.y;
		}
	}
}

// Kernel function, all the particles coordinate minus (displacement)
__global__ void Kernel_minusDispalcement(particle_t* dev_particles, float2 displacement)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < NUM_PARTICLES) {
		// Get particle to be updated by this thread.
		dev_particles[id].x -= displacement.x;
		dev_particles[id].y -= displacement.y;
	}
}

// Kernel function, adjust angle so that mean angle <\theta_0>=0
__global__ void Kernel_AdjustAngle(particle_t* dev_particles, float meanAngle)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < NUM_PARTICLES) {
		// Get particle to be updated by this thread.
		dev_particles[id].theta -= meanAngle;
	}
}

int main(int argc, char** argv)
{
	// Allocate host buffer to store particle States and Neighbours.
	particle_t* particles;
	cudaMallocHost(&particles, sizeof(particle_t)*NUM_PARTICLES);
	int* NeighbourList;
	cudaMallocHost(&NeighbourList, NUM_PARTICLES * sizeof(int) * 6);// 6 neighbours' index
	float* ElasticForce;
	cudaMallocHost(&ElasticForce, NUM_PARTICLES * sizeof(float) * 2);// Elastic Force matrix of size (NUM_PARTICLES,2)
	float2* particle0xy;
	cudaMallocHost(&particle0xy, sizeof(float2));// a copy of particle 0's x and y coordinate, used to make whole displacement
	float* direction_Noise;
	cudaMallocHost(&direction_Noise, NUM_PARTICLES * sizeof(float));// active Force Noise matrix of size (NUM_PARTICLES,1)
	float* vel_Noise;
	cudaMallocHost(&vel_Noise, NUM_PARTICLES * sizeof(float) * 2);// Velocity Noise matrix of size (NUM_PARTICLES,2)

	initTriangleGrid(particles, NeighbourList); // initialize triangle grid, allocate velocities and Active Forces

	// Allocate device memory for particles.
	particle_t *dev_particles;
	int	*dev_NeighbourList;
	float *dev_ElasticForce, *dev_vel_Noise, *dev_direction_Noise;
	float2 *dev_particle0xy;
	HANDLE_ERROR(cudaMalloc((void**)&dev_particles, NUM_PARTICLES * sizeof(particle_t)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_NeighbourList, NUM_PARTICLES * sizeof(int) * 6));
	HANDLE_ERROR(cudaMalloc((void**)&dev_ElasticForce, NUM_PARTICLES * sizeof(float) * 2));
	HANDLE_ERROR(cudaMalloc((void**)&dev_vel_Noise, NUM_PARTICLES * sizeof(float) * 2));
	HANDLE_ERROR(cudaMalloc((void**)&dev_direction_Noise, NUM_PARTICLES * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_particle0xy, sizeof(float2)));
	HANDLE_ERROR(cudaMemcpy(dev_particles, particles, NUM_PARTICLES * sizeof(particle_t), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_NeighbourList, NeighbourList, NUM_PARTICLES * sizeof(int) * 6, cudaMemcpyHostToDevice));

	// set up Random Number Generator for Noise
	random_device rd;
	mt19937 gen(rd()); // use Mersenne Twister algorithm
	normal_distribution<float> Normal(0, 1);

	// kernel function config
	int threadsPerBlock = SHARED_BUFFER_SIZE;
	dim3 gridSize(NUM_PARTICLES / threadsPerBlock);
	dim3 blockSize(threadsPerBlock);
	//cudaSetDevice(1);

	clock_t start = clock(), end;
	float cpu_time_used;
	int flag = 0;
	int nStep = 100; // 200000
	int recordFrequency = 1; //1000

	// write in the initial config
	string str = "dataFiles\\data" + to_string(flag) + ".txt"; ofstream mycout(str);
	mycout << "x	y	theta" << endl;
	for (int i = 0; i < NUM_PARTICLES; i++) {
		mycout << particles[i].x << "		" << particles[i].y << "		" << particles[i].theta<< endl;
	}
	mycout.close(); flag++;

	for (int step = 0; step < nStep; step++)
	{
		// update & transfer velocity noise
		for (int i = 0; i < NUM_PARTICLES; i++) {
			vel_Noise[i * 2 + 0] = Normal(gen); vel_Noise[i * 2 + 1] = Normal(gen);
			direction_Noise[i] = Normal(gen);
		}
		// transfer 2 noise matrix
		HANDLE_ERROR(cudaMemcpy(dev_vel_Noise, vel_Noise, NUM_PARTICLES * sizeof(float) * 2, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_direction_Noise, direction_Noise, NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));

		//// Call Kernel function. Update particles.
		Kerneltimestep_Overdamped << <gridSize, blockSize >> >(dev_particles, dev_NeighbourList, dev_ElasticForce, dev_vel_Noise, dev_direction_Noise, dev_particle0xy);

		cudaDeviceSynchronize();

		if (step % recordFrequency == 0) { // record every (recordFrequency) steps
			// displace all particles so that particle 0 is at the origin
			HANDLE_ERROR(cudaMemcpy(particle0xy, dev_particle0xy, sizeof(float2), cudaMemcpyDeviceToHost)); // Copy ONLY particle0's (x,y) coordinate back to host.
			float2 displacement = particle0xy[0];
			// make displacement with respect to particle 0
			Kernel_minusDispalcement << <gridSize, blockSize >> >(dev_particles, displacement);

			// Copy particle positions back to host.
			HANDLE_ERROR(cudaMemcpy(particles, dev_particles, NUM_PARTICLES*sizeof(particle_t), cudaMemcpyDeviceToHost));
			//HANDLE_ERROR(cudaMemcpy(ElasticForce, dev_ElasticForce, NUM_PARTICLES*sizeof(float) * 2, cudaMemcpyDeviceToHost));

			// write in data
			string str = "dataFiles\\data" + to_string(flag) + ".txt";
			ofstream mycout(str);
			mycout << "x	y	theta" << endl;
			for (int i = 0; i < NUM_PARTICLES; i++) {
				mycout << particles[i].x << "		" << particles[i].y << "		" << particles[i].theta << endl;
			}
			mycout.close();
			flag++;
			end = clock();
			cpu_time_used = ((float)(end - start)) / CLOCKS_PER_SEC;
			ofstream recordTimeAndProgress("Time and Progress.txt");
			recordTimeAndProgress << "time used: " << cpu_time_used / 60.0f << " mins" << endl;
			recordTimeAndProgress << "progress:" << floorf((float)step / (float)nStep*100.0f) << "%" << endl;
			recordTimeAndProgress << "time remaining:" << cpu_time_used*(nStep - step) / step / 60 << " mins" << endl;
			recordTimeAndProgress.close();

			//// adjust angle so that mean angle <\theta_0>=0
			//float angle = 0.0f;
			//for (int i = 0; i < NUM_PARTICLES; i++) {
			//	angle += particles[i].theta;
			//}
			//float total_Num = NUM_PARTICLES;
			//angle = angle / total_Num;
			////cout << "step="<< step<<", mean angle="<< angle << endl;
			//Kernel_AdjustAngle << <gridSize, blockSize >> >(dev_particles, angle);
		}

	}

	cudaFree(dev_particles);
	cudaFree(dev_NeighbourList);
	cudaFree(dev_ElasticForce);
	cudaFree(dev_vel_Noise);
	cudaFree(dev_direction_Noise);
	cudaFree(dev_particle0xy);

	return 0;
}
