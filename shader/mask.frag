#version 330 core
out vec4 outColor;
uniform vec3 maskColor;
void main() {
    outColor = vec4(maskColor, 1.0);
}
