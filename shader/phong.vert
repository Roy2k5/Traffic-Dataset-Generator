#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;

out vec3 fragNormal;
out vec2 fragTexCoord;
out vec3 fragPosition;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    mat4 modelView = view * model;
    vec4 viewPos = modelView * vec4(position, 1.0);
    fragPosition = viewPos.xyz;
    
    // Normal matrix to transform normals correctly
    mat3 normalMatrix = transpose(inverse(mat3(modelView)));
    fragNormal = normalize(normalMatrix * normal);
    
    fragTexCoord = texcoord;
    gl_Position = projection * viewPos;
}
