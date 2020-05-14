/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Bicubic texture filtering sample
    sgreen 6/2008

    This sample demonstrates how to efficiently implement bicubic texture
    filtering in CUDA.

    Bicubic filtering is a higher order interpolation method that produces
    smoother results than bilinear interpolation:
    http://en.wikipedia.org/wiki/Bicubic

    It requires reading a 4 x 4 pixel neighbourhood rather than the
    2 x 2 area required by bilinear filtering.

    Current graphics hardware doesn't support bicubic filtering natively,
    but it is possible to compose a bicubic filter using just 4 bilinear
    lookups by offsetting the sample position within each texel and weighting
    the samples correctly. The only disadvantage to this method is that the
    hardware only maintains 9-bits of filtering precision within each texel.

    See "Fast Third-Order Texture Filtering", Sigg & Hadwiger, GPU Gems 2:
    http://developer.nvidia.com/object/gpu_gems_2_home.html

    v1.1 - updated to include the brute force method using 16 texture lookups.
    v1.2 - added Catmull-Rom interpolation

    Example performance results from GeForce 8800 GTS:
    Bilinear     - 5500 MPixels/sec
    Bicubic      - 1400 MPixels/sec
    Fast Bicubic - 2100 MPixels/sec
*/

// OpenGL Graphics includes
#include <helper_gl.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else

#include <GL/freeglut.h>

#endif

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA system and GL includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Helper functions
#include <helper_functions.h> // CUDA SDK Helper functions
#include <helper_cuda.h>      // CUDA device initialization helper functions

typedef unsigned int uint;
typedef unsigned char uchar;

#define USE_BUFFER_TEX 0
#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 4;        // FPS limit for sampling
unsigned int frameCount = 0;
StopWatchInterface *timer = 0;
bool g_Verify = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define REFRESH_DELAY     10 //ms

static const char *title = "CUDA Particles";


uint width = 512, height = 512;
uint imageWidth, imageHeight;
dim3 blockSize(16, 16);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

//bool drawCurves = false;

GLuint pbo = 0;          // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
GLuint displayTex = 0;
GLuint bufferTex = 0;
GLuint fprog;                   // fragment program (shader)

float tx = 9.0f, ty = 10.0f;    // image translation
float scale = 1.0f / 16.0f;     // image scale

void display();

void initGLBuffers();

//void runBenchmark(int iterations);
void cleanup();

#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB
//#define GL_TEXTURE_TYPE GL_TEXTURE_2D

extern "C" void initGL(int *argc, char **argv);

extern "C" void onIdle();
extern "C" void setGravity(bool enabled);
extern "C" void setColourMode(int cMode);

extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output);

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float bspline_w0(float a) {
    return (1.0f / 6.0f) * (-a * a * a + 3.0f * a * a - 3.0f * a + 1.0f);
}

float bspline_w1(float a) {
    return (1.0f / 6.0f) * (3.0f * a * a * a - 6.0f * a * a + 4.0f);
}

float bspline_w2(float a) {
    return (1.0f / 6.0f) * (-3.0f * a * a * a + 3.0f * a * a + 3.0f * a + 1.0f);
}

__host__ __device__
float bspline_w3(float a) {
    return (1.0f / 6.0f) * (a * a * a);
}

void computeFPS() {
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit - 1) {
        g_Verify = true;
    }

    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "%s: %3.1f fps", title, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        sdkResetTimer(&timer);
    }
}

// display results using OpenGL (called by GLUT)
void display() {
    sdkStartTimer(&timer);

    // map PBO to get CUDA device pointer
    uchar4 *d_output;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &d_output, &num_bytes,
                                                         cuda_pbo_resource));
    render(imageWidth, imageHeight, blockSize, gridSize, d_output);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // Common display path
    {
        // display results
        glClear(GL_COLOR_BUFFER_BIT);

#if USE_BUFFER_TEX
        // display using buffer texture
        glBindTexture(GL_TEXTURE_BUFFER_EXT, bufferTex);
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fprog);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glProgramLocalParameterI4iNV(GL_FRAGMENT_PROGRAM_ARB, 0, width, 0, 0, 0);
#else
        // download image from PBO to OpenGL texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBindTexture(GL_TEXTURE_TYPE, displayTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_TYPE,
                        0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glEnable(GL_TEXTURE_TYPE);
#endif

        // draw textured quad
        glDisable(GL_DEPTH_TEST);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, (GLfloat) height);
        glVertex2f(0.0f, 0.0f);
        glTexCoord2f((GLfloat) width, (GLfloat) height);
        glVertex2f(1.0f, 0.0f);
        glTexCoord2f((GLfloat) width, 0.0f);
        glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(0.0f, 1.0f);
        glEnd();
        glDisable(GL_TEXTURE_TYPE);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    }

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);
    setGravity(false);
    computeFPS();
}

// GLUT callback functions
void timerEvent(int value) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}


void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {
        case 27:
            glutDestroyWindow(glutGetWindow());
            return;

        case '=':
        case '+':
            scale *= 0.5f;
            break;

        case '-':
            scale *= 2.0f;
            break;

        case 'g':
            setGravity(true);
            break;

        case 'r':
            scale = 1.0f;
            tx = ty = 0.0f;
            break;

        case 'd':
            printf("%f, %f, %f\n", tx, ty, scale);
            break;

        case 'b':
            //runBenchmark(500);
            break;

        default:
            break;
    }

    if (key >= '0' && key <= '5') {
        setColourMode(key);
    }

}

void reshape(int x, int y) {
    width = x;
    height = y;
    imageWidth = width;
    imageHeight = height;

    initGLBuffers();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup() {
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

    glDeleteBuffers(1, &pbo);

#if USE_BUFFER_TEX
    glDeleteTextures(1, &bufferTex);
    glDeleteProgramsARB(1, &fprog);
#else
    glDeleteTextures(1, &displayTex);
#endif

    sdkDeleteTimer(&timer);
}

int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initGLBuffers() {
    if (pbo) {
        // delete old buffer
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        glDeleteBuffers(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                                                 cudaGraphicsMapFlagsWriteDiscard));

#if USE_BUFFER_TEX

    // create buffer texture, attach to pbo
    if (bufferTex)
    {
        glDeleteTextures(1, &bufferTex);
    }

    glGenTextures(1, &bufferTex);
    glBindTexture(GL_TEXTURE_BUFFER_EXT, bufferTex);
    glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA8, pbo);
    glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
#else

    // create texture for display
    if (displayTex) {
        glDeleteTextures(1, &displayTex);
    }

    glGenTextures(1, &displayTex);
    glBindTexture(GL_TEXTURE_TYPE, displayTex);
    glTexImage2D(GL_TEXTURE_TYPE, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_TYPE, 0);
#endif

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

void mainMenu(int i) {
    keyboard(i, 0, 0);
}

#if USE_BUFFER_TEX
// fragment program for reading from buffer texture
static const char *shaderCode =
"!!NVfp4.0\n"
"INT PARAM width = program.local[0];\n"
"INT TEMP index;\n"
"FLR.S index, fragment.texcoord;\n"
"MAD.S index.x, index.y, width, index.x;\n" // compute 1D index from 2D coords
"TXF result.color, index.x, texture[0], BUFFER;\n"
"END";
#endif

GLuint compileASMShader(GLenum program_type, const char *code) {
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int) error_pos, error_string);
        return 0;
    }

    return program_id;
}

void initialize(int argc, char **argv) {
    initGL(&argc, argv);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **) argv);

    // get number of SMs on this GPU
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

    // Create the timer (for fps measurement)
    sdkCreateTimer(&timer);

    initGLBuffers();

#if USE_BUFFER_TEX
    fprog = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shaderCode);

    if (!fprog)
    {
        exit(EXIT_SUCCESS);
    }

#endif
}

void initGL(int *argc, char **argv) {
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Particles");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
//    glutMouseFunc(mouse);
//    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(onIdle);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    if (!isGLVersionSupported(2, 0) ||
        !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(EXIT_FAILURE);
    }

#if USE_BUFFER_TEX

    if (!areGLExtensionsSupported("GL_EXT_texture_buffer_object"))
    {
        fprintf(stderr, "OpenGL extension: GL_EXT_texture_buffer_object missing.\n");
        exit(EXIT_FAILURE);
    }

    if (!areGLExtensionsSupported("GL_NV_gpu_program4"))
    {
        fprintf(stderr, "OpenGL extension: GL_NV_gpu_program4 missing.\n");
        exit(EXIT_FAILURE);
    }

#endif
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    pArgc = &argc;
    pArgv = argv;

    // parse arguments
    char *filename;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("Starting %s\n", title);

    initialize(argc, argv);
    glutMainLoop();

    exit(EXIT_SUCCESS);
}
