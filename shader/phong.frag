#version 330 core
in vec3 fragNormal;
in vec2 fragTexCoord;
in vec3 fragPosition;

out vec4 outColor;

uniform sampler2D diffuseTex;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform bool useTexture;
uniform vec3 baseColor;
uniform vec3 colorTint;   // (1,1,1) = no tint; car color for body parts

void main() {
    float ambientStrength = 0.35;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec4 texColor = useTexture ? texture(diffuseTex, fragTexCoord) : vec4(baseColor, 1.0);
    vec3 result = (ambient + diffuse) * texColor.rgb * colorTint;

    outColor = vec4(result, texColor.a);
}
