#version 330 core
in vec3 fragPosition;
out vec4 outColor;
uniform float zNear;
uniform float zFar;

void main() {
    float z = gl_FragCoord.z * 2.0 - 1.0;
    float linearDepth = (2.0 * zNear * zFar) / (zFar + zNear - z * (zFar - zNear));
    // Normalise to [0,1] within [zNear, zFar] — bright = far
    float nd = clamp((linearDepth - zNear) / (zFar - zNear), 0.0, 1.0);
    outColor = vec4(vec3(nd), 1.0);
}
