import OpenGL.GL as GL
import numpy as np
import colorsys
from libs.shader import Shader
from libs.transform import translate, rotate, scale

# Pre-built palette: index 0 = background black, 1..N = vivid distinct hues
_PALETTE_SIZE = 128
MASK_PALETTE = [(0, 0, 0)]   # id 0 = background
for _i in range(1, _PALETTE_SIZE):
    _h = (_i * 0.618033988749895) % 1.0
    _r, _g, _b = colorsys.hsv_to_rgb(_h, 0.95, 0.95)
    MASK_PALETTE.append((int(_r * 255), int(_g * 255), int(_b * 255)))


def id_to_color_f(obj_id):
    """Return (r,g,b) float [0..1] for a given object id."""
    r, g, b = MASK_PALETTE[obj_id % _PALETTE_SIZE]
    return r / 255., g / 255., b / 255.


class Renderer:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.fbo = self.color_tex = self.depth_rbo = None
        self.rgb_shader = self.depth_shader = self.mask_shader = None
        self.bg_color = (0.5, 0.7, 1.0)

    def init_gl(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        # Render both sides of every polygon — OBJ assets have mixed winding
        # orders so culling would hide valid faces on buildings and car body.
        GL.glDisable(GL.GL_CULL_FACE)
        self.rgb_shader   = Shader('shader/phong.vert', 'shader/phong.frag')
        self.depth_shader = Shader('shader/phong.vert', 'shader/depth.frag')
        self.mask_shader  = Shader('shader/phong.vert', 'shader/mask.frag')
        self._create_fbo(self.width, self.height)

    def resize(self, w, h):
        if w <= 0 or h <= 0:
            return
        self.width, self.height = w, h
        if self.fbo is not None:
            self._create_fbo(w, h)

    def _create_fbo(self, w, h):
        if self.fbo is not None:
            GL.glDeleteFramebuffers(1, [self.fbo])
            GL.glDeleteTextures([self.color_tex])
            GL.glDeleteRenderbuffers(1, [self.depth_rbo])

        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        self.color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self.color_tex, 0)
        self.depth_rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth_rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8, w, h)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                     GL.GL_RENDERBUFFER, self.depth_rbo)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def render(self, scene_objects, camera, pass_type='rgb',
               offscreen=False, default_fbo=None, vp_width=None, vp_height=None):
        if offscreen:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            vp_w, vp_h = self.width, self.height
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,
                                 default_fbo if default_fbo is not None else 0)
            vp_w = vp_width  or self.width
            vp_h = vp_height or self.height

        GL.glViewport(0, 0, vp_w, vp_h)

        if pass_type == 'mask':
            GL.glClearColor(0, 0, 0, 1)
        else:
            r, g, b = self.bg_color
            GL.glClearColor(r, g, b, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        shader_obj = {'rgb': self.rgb_shader,
                      'depth': self.depth_shader,
                      'mask': self.mask_shader}[pass_type]
        shader = shader_obj.render_idx
        GL.glUseProgram(shader)

        # ── Matrices ─────────────────────────────────────────────────────────
        proj = camera.projection_matrix([vp_w, vp_h])
        view = camera.view_matrix()
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, "projection"),
                              1, GL.GL_TRUE, proj)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, "view"),
                              1, GL.GL_TRUE, view)

        if pass_type == 'rgb':
            GL.glUniform3f(GL.glGetUniformLocation(shader, "lightPos"),   0, 30, 15)
            GL.glUniform3f(GL.glGetUniformLocation(shader, "lightColor"), 1,  1,  1)
        elif pass_type == 'depth':
            # Use distance-relative near/far for good gradient across the scene
            z_near = camera.distance * 0.05
            z_far  = camera.distance * 2.5
            GL.glUniform1f(GL.glGetUniformLocation(shader, "zNear"), z_near)
            GL.glUniform1f(GL.glGetUniformLocation(shader, "zFar"),  z_far)

        model_loc = GL.glGetUniformLocation(shader, "model")

        for obj in scene_objects:
            pos = obj['position']
            rot = obj['rotation']
            scl = obj['scale']

            m = (translate(pos)
                 @ rotate((1, 0, 0), rot[0] * 180 / np.pi)
                 @ rotate((0, 1, 0), rot[1] * 180 / np.pi)
                 @ rotate((0, 0, 1), rot[2] * 180 / np.pi)
                 @ scale(scl))
            GL.glUniformMatrix4fv(model_loc, 1, GL.GL_TRUE, m)

            if pass_type == 'mask':
                c = id_to_color_f(obj['id'])
                GL.glUniform3f(GL.glGetUniformLocation(shader, "maskColor"), *c)

            color_override = obj.get('color') if pass_type == 'rgb' else None
            wheel_angle    = obj.get('wheel_angle', 0.0)
            from scene.car_mesh import CarMesh
            if isinstance(obj['mesh'], CarMesh):
                obj['mesh'].draw(shader, color_override=color_override,
                                 model_matrix=m, wheel_angle=wheel_angle)
            else:
                obj['mesh'].draw(shader, color_override=color_override)

        GL.glUseProgram(0)

        if offscreen:
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            px = GL.glReadPixels(0, 0, self.width, self.height,
                                 GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            img = np.frombuffer(px, dtype=np.uint8).reshape(
                (self.height, self.width, 3))
            return np.flipud(img)
        return None
