
#include <cmath>
#include <cstdio>

#define CUTOFF 4 // Cutoff in standard deviations for evaluating the Gaussian exponent

extern "C" {

    void peak_dist(
        int nb, int nx, int ny, int nz, float *dist,
        int *N_atom, float *pos,
        float xyz_start[3], float xyz_step[3], float std
    ) {

        int nxyz = nx * ny * nz;
        float cov_inv = 1 / (std * std);
        float cutoff2 = CUTOFF * CUTOFF * std * std;

        float pi = 2 * acos(0.0);
        float denom = sqrt(2*pi) * std;
        float prefactor = 1 / (denom * denom * denom);

        int ind = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    float grid_xyz[3] = {
                        xyz_start[0] + i * xyz_step[0],
                        xyz_start[1] + j * xyz_step[1],
                        xyz_start[2] + k * xyz_step[2]
                    };
                    int ip0 = 0;
                    for (int b = 0; b < nb; b++) {
                        float v = 0;
                        int ip = 0;
                        for (; ip < N_atom[b]; ip++) {
                            float atom_xyz[3] = {pos[3*(ip0 + ip)], pos[3*(ip0 + ip)+1], pos[3*(ip0 + ip)+2]};
                            float dp[3] = {
                                (grid_xyz[0] - atom_xyz[0]),
                                (grid_xyz[1] - atom_xyz[1]),
                                (grid_xyz[2] - atom_xyz[2])
                            };
                            float dp2 = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
                            if (dp2 < cutoff2) {
                                v += exp(-0.5 * dp2 * cov_inv);
                            }
                        }
                        v *= prefactor;
                        ip0 += ip;
                        dist[b * nxyz + ind] = v;
                    }
                    ind++;
                }
            }
        }

    }

}
