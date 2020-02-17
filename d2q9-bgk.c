/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
// #define DEBUG

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct{
  float *speeds0;
  float *speedsN;
  float *speedsS;
  float *speedsW;
  float *speedsE;
  float *speedsNW;
  float *speedsNE;
  float *speedsSW;
  float *speedsSE;
}t_speed_arr;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed_arr* __restrict__ cells, t_speed_arr* __restrict__ tmp_cells, int* __restrict__ obstacles);
int accelerate_flow(const t_param params, t_speed_arr* __restrict__ cells, int* __restrict__ obstacles);
float propagate(const t_param params, t_speed_arr* __restrict__ cells, t_speed_arr* __restrict__ tmp_cells,int* __restrict__ obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_arr* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_arr* __restrict__ cells, int* __restrict__ obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_arr* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }


  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
  //initialise temp cells array
  float* speeds0 = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* speedsN = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* speedsS = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* speedsW = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* speedsE = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* speedsNW = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* speedsNE = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* speedsSW = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* speedsSE = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);

  float* tmp_speeds0 = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* tmp_speedsN = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* tmp_speedsS = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* tmp_speedsW = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* tmp_speedsE = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* tmp_speedsNW = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* tmp_speedsNE = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* tmp_speedsSW = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  float* tmp_speedsSE = (float*)_mm_malloc(sizeof(float) * params.ny * params.nx,64);
  __assume_aligned(tmp_speeds0, 64);
  __assume_aligned(tmp_speedsN, 64);
  __assume_aligned(tmp_speedsS, 64);
  __assume_aligned(tmp_speedsE, 64);
  __assume_aligned(tmp_speedsW, 64);
  __assume_aligned(tmp_speedsNW, 64);
  __assume_aligned(tmp_speedsNE, 64);
  __assume_aligned(tmp_speedsSW, 64);
  __assume_aligned(tmp_speedsSE, 64);

  __assume_aligned(speeds0, 64);
  __assume_aligned(speedsN, 64);
  __assume_aligned(speedsS, 64);
  __assume_aligned(speedsE, 64);
  __assume_aligned(speedsW, 64);
  __assume_aligned(speedsNW, 64);
  __assume_aligned(speedsNE, 64);
  __assume_aligned(speedsSW, 64);
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int index =ii + jj*params.nx;
      speeds0[index] = cells[index].speeds[0];
      tmp_speeds0[index] = tmp_cells[index].speeds[0];

      speedsN[index] = cells[index].speeds[2];
      tmp_speedsN[index] = tmp_cells[index].speeds[2];

      speedsS[index] = cells[index].speeds[4];
      tmp_speedsS[index] = tmp_cells[index].speeds[4];

      speedsW[index] = cells[index].speeds[3];
      tmp_speedsW[index] = tmp_cells[index].speeds[3];

      speedsE[index] = cells[index].speeds[1];
      tmp_speedsE[index] = tmp_cells[index].speeds[1];

      speedsNW[index] = cells[index].speeds[6];
      tmp_speedsNW[index] = tmp_cells[index].speeds[6];

      speedsNE[index] = cells[index].speeds[5];
      tmp_speedsNE[index] = tmp_cells[index].speeds[5];

      speedsSW[index] = cells[index].speeds[7];
      tmp_speedsSW[index] = tmp_cells[index].speeds[7];

      speedsSE[index] = cells[index].speeds[8];
      tmp_speedsSE[index] = tmp_cells[index].speeds[8];
    }
  }
  t_speed_arr tmp_cells_arr = {tmp_speeds0,tmp_speedsN,tmp_speedsS,tmp_speedsW,tmp_speedsE,
    tmp_speedsNW,tmp_speedsNE,tmp_speedsSW,tmp_speedsSE};

  t_speed_arr cells_arr = {speeds0,speedsN,speedsS,speedsW,speedsE,
    speedsNW,speedsNE,speedsSW,speedsSE};
    t_speed_arr* cells_arr_ptr = &cells_arr;
    t_speed_arr* tmp_cells_arr_ptr = &tmp_cells_arr;
  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);



  for (int tt = 0; tt < params.maxIters; tt++)
  {
    av_vels[tt] = timestep(params, cells_arr_ptr, tmp_cells_arr_ptr, obstacles);
    // timestep(params, cells_arr_ptr, tmp_cells_arr_ptr, obstacles);
    // av_vels[tt] = av_velocity(params, tmp_cells_arr_ptr, obstacles);
    t_speed_arr* p = cells_arr_ptr;
    cells_arr_ptr = tmp_cells_arr_ptr;
    tmp_cells_arr_ptr = p;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, &cells_arr));
#endif
  }


  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, &cells_arr, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  //put ASSUME_ALIGNED HERE

  //copy values back into cells
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    #pragma vector aligned

    for (int ii = 0; ii < params.nx; ii++)
    {
      int index =ii + jj*params.nx;
      cells[index].speeds[0] = cells_arr.speeds0[index];
      cells[index].speeds[1] = cells_arr.speedsE[index];
      cells[index].speeds[2] = cells_arr.speedsN[index];
      cells[index].speeds[3] = cells_arr.speedsW[index];
      cells[index].speeds[4] = cells_arr.speedsS[index];
      cells[index].speeds[5] = cells_arr.speedsNE[index];
      cells[index].speeds[6] = cells_arr.speedsNW[index];
      cells[index].speeds[7] = cells_arr.speedsSW[index];
      cells[index].speeds[8] = cells_arr.speedsSE[index];
    }
  }
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed_arr* __restrict__ cells_arr, t_speed_arr* __restrict__ tmp_cells_arr, int* __restrict__ obstacles)
{

  accelerate_flow(params, cells_arr, obstacles);

  // rebound(params, cells, tmp_cells, obstacles);
 // rebound_and_collision(params, cells_arr, tmp_cells_arr, obstacles);
  return propagate(params, cells_arr, tmp_cells_arr,obstacles);;
}

int accelerate_flow(const t_param params, t_speed_arr* __restrict__ cells, int* __restrict__ obstacles)
{

  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
 const int jj = params.ny - 2;

 //PUT OPENMP FOR Pragma here?
 __assume_aligned(cells->speeds0, 64);
 __assume_aligned(cells->speedsN, 64);
 __assume_aligned(cells->speedsS, 64);
 __assume_aligned(cells->speedsE, 64);
 __assume_aligned(cells->speedsW, 64);
 __assume_aligned(cells->speedsNW, 64);
 __assume_aligned(cells->speedsNE, 64);
 __assume_aligned(cells->speedsSW, 64);
 __assume_aligned(cells->speedsSE, 64);
 __assume((params.nx)%128==0);
 #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    int index = ii + jj*params.nx;

    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[index]
        && (cells->speedsW[index] - w1) > 0.f
        && (cells->speedsNW[index] - w2) > 0.f
        && (cells->speedsSW[index] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speedsE[index] += w1;
      cells->speedsNE[index] += w2;
      cells->speedsSE[index] += w2;
      /* decrease 'west-side' densities */
      cells->speedsW[index] -= w1;
      cells->speedsNW[index] -= w2;
      cells->speedsSW[index] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float propagate(const t_param params, t_speed_arr* __restrict__ cells, t_speed_arr* __restrict__ tmp_cells,int* __restrict__ obstacles)
{



  //propagation needs neighbouring cells

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  const float b = (2* c_sq * c_sq);
  const float d = 2 * c_sq;

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;




  float *d_equ = (float*)_mm_malloc(sizeof(float) * NSPEEDS,64);
  float *u     = (float*)_mm_malloc(sizeof(float) * NSPEEDS,64);

  __assume_aligned(d_equ,64);
  __assume_aligned(u,64);
  /* loop over _all_ cells */
  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speedsN, 64);
  __assume_aligned(cells->speedsS, 64);
  __assume_aligned(cells->speedsE, 64);
  __assume_aligned(cells->speedsW, 64);
  __assume_aligned(cells->speedsNW, 64);
  __assume_aligned(cells->speedsNE, 64);
  __assume_aligned(cells->speedsSW, 64);
  __assume_aligned(cells->speedsSE, 64);

  __assume_aligned(tmp_cells->speeds0, 64);
  __assume_aligned(tmp_cells->speedsN, 64);
  __assume_aligned(tmp_cells->speedsS, 64);
  __assume_aligned(tmp_cells->speedsE, 64);
  __assume_aligned(tmp_cells->speedsW, 64);
  __assume_aligned(tmp_cells->speedsNW, 64);
  __assume_aligned(tmp_cells->speedsNE, 64);
  __assume_aligned(tmp_cells->speedsSW, 64);
  __assume_aligned(tmp_cells->speedsSE, 64);
    // __assume_aligned(cells,64);
    // __assume_aligned(tmp_cells,64);
  __assume((params.nx)%128==0);
  __assume((params.ny)%128==0);

  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {

      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      int index= ii + jj*params.nx;
      if(obstacles[index]){
          tmp_cells->speedsE[index] = cells->speedsW[x_e + jj*params.nx];
          tmp_cells->speedsN[index] = cells->speedsS[ii + y_n*params.nx];
          tmp_cells->speedsW[index] = cells->speedsE[x_w + jj*params.nx];
          tmp_cells->speedsS[index] = cells->speedsN[ii + y_s*params.nx];
          tmp_cells->speedsNE[index] = cells->speedsSW[x_e + y_n*params.nx];
          tmp_cells->speedsNW[index] = cells->speedsSE[x_w + y_n*params.nx];
          tmp_cells->speedsSW[index] = cells->speedsNE[x_w + y_s*params.nx];
          tmp_cells->speedsSE[index] = cells->speedsNW[x_e + y_s*params.nx];
      }else{
          tmp_cells->speeds0[index] = cells->speeds0[ii+jj*params.nx]; /* central cell, no movement */
          tmp_cells->speedsE[index] = cells->speedsE[x_w + jj*params.nx]; /* east */
          tmp_cells->speedsN[index] = cells->speedsN[ii + y_s*params.nx]; /* north */
          tmp_cells->speedsW[index] = cells->speedsW[x_e + jj*params.nx]; /* west */
          tmp_cells->speedsS[index] = cells->speedsS[ii + y_n*params.nx]; /* south */
          tmp_cells->speedsNE[index] = cells->speedsNE[x_w + y_s*params.nx]; /* north-east */
          tmp_cells->speedsNW[index] = cells->speedsNW[x_e + y_s*params.nx]; /* north-west */
          tmp_cells->speedsSW[index] = cells->speedsSW[x_e + y_n*params.nx]; /* south-west */
          tmp_cells->speedsSE[index] = cells->speedsSE[x_w + y_n*params.nx]; /* south-east */
          // /* compute local density total */
          float local_density = 0.f;
          local_density += tmp_cells->speeds0[index];
          local_density += tmp_cells->speedsN[index];
          local_density += tmp_cells->speedsS[index];
          local_density += tmp_cells->speedsW[index];
          local_density += tmp_cells->speedsE[index];
          local_density += tmp_cells->speedsNW[index];
          local_density += tmp_cells->speedsNE[index];
          local_density += tmp_cells->speedsSW[index];
          local_density += tmp_cells->speedsSE[index];
          /* compute x velocity component */
          float u_x = (tmp_cells->speedsE[index]
                        + tmp_cells->speedsNE[index]
                        + tmp_cells->speedsSE[index]
                        - (tmp_cells->speedsW[index]
                           + tmp_cells->speedsNW[index]
                           + tmp_cells->speedsSW[index])  )
                       / local_density;
          /* compute y velocity component */
          float u_y = (tmp_cells->speedsN[index]
                        + tmp_cells->speedsNE[index]
                        + tmp_cells->speedsNW[index]
                        - (tmp_cells->speedsS[index]
                           + tmp_cells->speedsSW[index]
                           + tmp_cells->speedsSE[index]) )
                       / local_density;

          /* velocity squared */
          float u_sq = u_x * u_x + u_y * u_y;

          /* directional velocity components */
          // float u[NSPEEDS];
          u[1] =   u_x;        /* east */
          u[2] =         u_y;  /* north */
          u[3] = - u_x;        /* west */
          u[4] =       - u_y;  /* south */
          u[5] =   u_x + u_y;  /* north-east */
          u[6] = - u_x + u_y;  /* north-west */
          u[7] = - u_x - u_y;  /* south-west */
          u[8] =   u_x - u_y;  /* south-east */

           float e = c_sq * u_sq;
            /* equilibrium densities */

            /* zero velocity density: weight w0 */
            d_equ[0] = w0 * local_density
                       * (1.f - u_sq/(2.f * c_sq) );

            d_equ[1] = w1 * local_density * ((b + d * u[1] + u[1] * u[1] - e) /b);
            d_equ[2] = w1 * local_density * ((b + d * u[2] + u[2] * u[2] - e) /b);
            d_equ[3] = w1 * local_density * ((b + d * u[3] + u[3] * u[3] - e) /b);
            d_equ[4] = w1 * local_density * ((b + d * u[4] + u[4] * u[4] - e) /b);
            /* diagonal speeds: weight w2 */
            d_equ[5] = w2 * local_density * ((b + d * u[5] + u[5] * u[5] - e) /b);
            d_equ[6] = w2 * local_density * ((b + d * u[6] + u[6] * u[6] - e) /b);
            d_equ[7] = w2 * local_density * ((b + d * u[7] + u[7] * u[7] - e) /b);
            d_equ[8] = w2 * local_density * ((b + d * u[8] + u[8] * u[8] - e) /b);

          /* relaxation step */

          tmp_cells->speeds0[index] = tmp_cells->speeds0[index] + params.omega * (d_equ[0] - tmp_cells->speeds0[index]);
          tmp_cells->speedsE[index] = tmp_cells->speedsE[index] + params.omega * (d_equ[1] - tmp_cells->speedsE[index]);
          tmp_cells->speedsN[index] = tmp_cells->speedsN[index] + params.omega * (d_equ[2] - tmp_cells->speedsN[index]);
          tmp_cells->speedsW[index] = tmp_cells->speedsW[index] + params.omega * (d_equ[3] - tmp_cells->speedsW[index]);
          tmp_cells->speedsS[index] = tmp_cells->speedsS[index] + params.omega * (d_equ[4] - tmp_cells->speedsS[index]);
          tmp_cells->speedsNE[index] = tmp_cells->speedsNE[index] + params.omega * (d_equ[5] - tmp_cells->speedsNE[index]);
          tmp_cells->speedsNW[index] = tmp_cells->speedsNW[index] + params.omega * (d_equ[6] - tmp_cells->speedsNW[index]);
          tmp_cells->speedsSW[index] = tmp_cells->speedsSW[index] + params.omega * (d_equ[7] - tmp_cells->speedsSW[index]);
          tmp_cells->speedsSE[index] = tmp_cells->speedsSE[index] + params.omega * (d_equ[8] - tmp_cells->speedsSE[index]);

          /* accumulate the norm of x- and y- velocity components */
          tot_u += sqrtf(u_sq);
          /* increase counter of inspected cells */
          ++tot_cells;
      }
    }
  }

  return tot_u/(float) tot_cells;
}


float av_velocity(const t_param params, t_speed_arr* __restrict__ cells, int* __restrict__ obstacles)
{

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speedsN, 64);
  __assume_aligned(cells->speedsS, 64);
  __assume_aligned(cells->speedsE, 64);
  __assume_aligned(cells->speedsW, 64);
  __assume_aligned(cells->speedsNW, 64);
  __assume_aligned(cells->speedsNE, 64);
  __assume_aligned(cells->speedsSW, 64);
  __assume_aligned(cells->speedsSE, 64);
  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    #pragma vector aligned
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        int index = ii + jj*params.nx;
        local_density += cells->speeds0[index];
        local_density += cells->speedsN[index];
        local_density += cells->speedsS[index];
        local_density += cells->speedsW[index];
        local_density += cells->speedsE[index];
        local_density += cells->speedsNW[index];
        local_density += cells->speedsNE[index];
        local_density += cells->speedsSW[index];
        local_density += cells->speedsSE[index];
        /* compute x velocity component */
        float u_x = (cells->speedsE[index]
                      + cells->speedsNE[index]
                      + cells->speedsSE[index]
                      - (cells->speedsW[index]
                         + cells->speedsNW[index]
                         + cells->speedsSW[index])  )
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speedsN[index]
                      + cells->speedsNE[index]
                      + cells->speedsNW[index]
                      - (cells->speedsS[index]
                         + cells->speedsSW[index]
                         + cells->speedsSE[index]) )
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_arr* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed_arr* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int index = ii+jj*params.nx;
      total += cells->speeds0[index];
      total += cells->speedsN[index];
      total += cells->speedsS[index];
      total += cells->speedsW[index];
      total += cells->speedsE[index];
      total += cells->speedsNW[index];
      total += cells->speedsNE[index];
      total += cells->speedsSW[index];
      total += cells->speedsSE[index];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
