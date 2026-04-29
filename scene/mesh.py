import pywavefront
import OpenGL.GL as GL
import numpy as np
from PIL import Image
import os

GLASS_KEYWORDS = ['glass', 'windscreen', 'window', 'transparent', 'lens', 'mirror']

class Mesh:
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_dir = os.path.dirname(os.path.abspath(file_path))
        self.scene = pywavefront.Wavefront(file_path, collect_faces=True, create_materials=True)
        self.meshes = []
        self._load_gl_data()

    def _load_gl_data(self):
        for name, material in self.scene.materials.items():
            vertex_format = material.vertex_format
            vertices = material.vertices
            if not vertices:
                continue

            vbo = GL.glGenBuffers(1)
            vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)

            vertex_data = np.array(vertices, dtype=np.float32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL.GL_STATIC_DRAW)

            has_tex  = 'T2F' in vertex_format
            has_norm = 'N3F' in vertex_format
            stride = (2 if has_tex else 0) + (3 if has_norm else 0) + 3
            stride_bytes = stride * 4

            offset = 0
            if has_tex:
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride_bytes, GL.ctypes.c_void_p(offset))
                offset += 8
            if has_norm:
                GL.glEnableVertexAttribArray(2)
                GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, stride_bytes, GL.ctypes.c_void_p(offset))
                offset += 12
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride_bytes, GL.ctypes.c_void_p(offset))
            GL.glBindVertexArray(0)

            texture_id = 0
            if material.texture:
                texture_id = self._load_texture(material.texture.path)

            diffuse = list(material.diffuse[:3]) if material.diffuse else [0.8, 0.8, 0.8]
            is_glass = any(kw in name.lower() for kw in GLASS_KEYWORDS)

            self.meshes.append({
                'vao': vao,
                'vbo': vbo,
                'vertex_count': len(vertex_data) // stride,
                'texture_id': texture_id,
                'diffuse': diffuse,
                'material_name': name,
                'is_glass': is_glass,
            })

    def _load_texture(self, tex_name):
        for candidate in [tex_name,
                          os.path.join(self.base_dir, tex_name),
                          os.path.join(self.base_dir, os.path.basename(tex_name))]:
            if os.path.exists(candidate):
                try:
                    img = Image.open(candidate).convert('RGBA')
                    img_data = np.array(list(img.getdata()), np.uint8)
                    tid = GL.glGenTextures(1)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height,
                                   0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                    return tid
                except Exception:
                    pass
        return 0

    def draw(self, shader, color_override=None):
        """Draw all sub-meshes. color_override=(r,g,b) tints body (non-glass) parts."""
        tint_loc       = GL.glGetUniformLocation(shader, "colorTint")
        use_tex_loc    = GL.glGetUniformLocation(shader, "useTexture")
        base_color_loc = GL.glGetUniformLocation(shader, "baseColor")

        for mesh in self.meshes:
            GL.glBindVertexArray(mesh['vao'])

            # Default tint = white (no effect)
            GL.glUniform3f(tint_loc, 1.0, 1.0, 1.0)

            if mesh['texture_id'] != 0:
                GL.glUniform1i(use_tex_loc, 1)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, mesh['texture_id'])
                GL.glUniform1i(GL.glGetUniformLocation(shader, "diffuseTex"), 0)
                # Tint textured non-glass surfaces with car color
                if color_override and not mesh['is_glass']:
                    GL.glUniform3f(tint_loc, *color_override)
            else:
                GL.glUniform1i(use_tex_loc, 0)
                if color_override and not mesh['is_glass']:
                    GL.glUniform3f(base_color_loc, *color_override)
                else:
                    GL.glUniform3f(base_color_loc, *mesh['diffuse'])

            GL.glDrawArrays(GL.GL_TRIANGLES, 0, mesh['vertex_count'])
            GL.glBindVertexArray(0)
