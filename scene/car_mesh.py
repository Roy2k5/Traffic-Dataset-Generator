"""CarMesh: OBJ parser with per-group wheel rotation support."""
import OpenGL.GL as GL
import numpy as np
import os
from PIL import Image

GLASS_KW   = ['glass', 'windscreen', 'window', 'transparent']
ROTATE_KW  = ['wheel.', 'bolt.', 'tyre', 'brakedisc', 'wheel_']
CENTER_KW  = ['wheelcentre', 'wheelcenter']


class CarMesh:
    def __init__(self, file_path):
        self.base_dir  = os.path.dirname(os.path.abspath(file_path))
        self._mtl_kd   = {}   # mat -> [r,g,b]
        self._mtl_tex  = {}   # mat -> texture path
        self._tex_cache = {}  # path -> GL tex id
        self._static   = []   # list of group dicts (no rotation)
        self._rotating = []   # list of group dicts (wheel rotation)
        self._parse(file_path)

    # ── OBJ Parse ─────────────────────────────────────────────────────────────
    def _parse(self, path):
        raw_v, raw_vt, raw_vn = [], [], []
        groups   = {}   # name -> {'tris':[], 'mat':str, 'pos':[]}
        cur, mat = 'default', 'default'

        with open(path, errors='replace') as f:
            for line in f:
                p = line.split()
                if not p: continue
                c = p[0]
                if   c == 'mtllib' and len(p)>1:
                    self._parse_mtl(os.path.join(self.base_dir, p[1]))
                elif c == 'v'  and len(p)>=4:
                    raw_v.append((float(p[1]),float(p[2]),float(p[3])))
                elif c == 'vt' and len(p)>=3:
                    raw_vt.append((float(p[1]),float(p[2])))
                elif c == 'vn' and len(p)>=4:
                    raw_vn.append((float(p[1]),float(p[2]),float(p[3])))
                elif c in ('g','o') and len(p)>1:
                    cur = p[1]
                    if cur not in groups:
                        groups[cur] = {'tris':[], 'mat':mat, 'pos':[]}
                elif c == 'usemtl' and len(p)>1:
                    mat = p[1]
                    if cur not in groups:
                        groups[cur] = {'tris':[], 'mat':mat, 'pos':[]}
                    else:
                        groups[cur]['mat'] = mat
                elif c == 'f' and len(p)>=4:
                    face = []
                    for tok in p[1:]:
                        ix = tok.split('/')
                        vi = int(ix[0])-1
                        ti = int(ix[1])-1 if len(ix)>1 and ix[1] else -1
                        ni = int(ix[2])-1 if len(ix)>2 and ix[2] else -1
                        face.append((vi,ti,ni))
                    if cur not in groups:
                        groups[cur] = {'tris':[], 'mat':mat, 'pos':[]}
                    for i in range(1, len(face)-1):
                        groups[cur]['tris'].append([face[0],face[i],face[i+1]])

        # collect positions per group for wheel-centre detection
        for g in groups.values():
            for tri in g['tris']:
                for (vi,_,__) in tri:
                    if 0 <= vi < len(raw_v):
                        g['pos'].append(raw_v[vi])

        # find wheel centres → sorted by (x, z) → 4 wheels FL FR BL BR
        centres = []
        for name, g in groups.items():
            if any(kw in name.lower() for kw in CENTER_KW) and g['pos']:
                xs=[p[0] for p in g['pos']]; ys=[p[1] for p in g['pos']]; zs=[p[2] for p in g['pos']]
                centres.append((sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)))
        centres.sort(key=lambda c: (round(c[0],1), c[2]))

        # build VAOs
        for name, g in groups.items():
            if not g['tris']: continue
            vd = self._build_vao(name, g, raw_v, raw_vt, raw_vn)
            if vd is None: continue
            nl = name.lower()
            if any(kw in nl for kw in ROTATE_KW) and centres:
                pos = g['pos']
                if pos:
                    gx = sum(p[0] for p in pos)/len(pos)
                    gz = sum(p[2] for p in pos)/len(pos)
                    ci = min(range(len(centres)),
                             key=lambda i: (centres[i][0]-gx)**2+(centres[i][2]-gz)**2)
                    vd['wheel_center'] = centres[ci]
                vd['rotating'] = True
                self._rotating.append(vd)
            else:
                vd['rotating'] = False
                self._static.append(vd)

    def _build_vao(self, name, g, raw_v, raw_vt, raw_vn):
        data = []
        for tri in g['tris']:
            # Skip whole triangle if any vertex index is out of range
            if any(not (0 <= vi < len(raw_v)) for (vi,_,__) in tri):
                continue
            for (vi,ti,ni) in tri:
                x,y,z = raw_v[vi]
                nx,ny,nz = raw_vn[ni] if 0<=ni<len(raw_vn) else (0,1,0)
                u,v_ = raw_vt[ti] if 0<=ti<len(raw_vt) else (0,0)
                data.extend([u,v_,nx,ny,nz,x,y,z])
        if not data: return None
        arr = np.array(data, dtype=np.float32)
        vbo = GL.glGenBuffers(1); vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
        s = 8*4
        GL.glEnableVertexAttribArray(1); GL.glVertexAttribPointer(1,2,GL.GL_FLOAT,False,s,GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(2); GL.glVertexAttribPointer(2,3,GL.GL_FLOAT,False,s,GL.ctypes.c_void_p(8))
        GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,False,s,GL.ctypes.c_void_p(20))
        GL.glBindVertexArray(0)
        mat = g['mat']
        tex_id = self._load_tex(self._mtl_tex.get(mat,''))
        return {'vao':vao,'vbo':vbo,'n':len(arr)//8,
                'tex':tex_id,'kd':self._mtl_kd.get(mat,[0.8,0.8,0.8]),
                'is_glass':any(kw in name.lower() for kw in GLASS_KW),
                'wheel_center':(0,0,0)}

    def _parse_mtl(self, path):
        if not os.path.exists(path): return
        cur = None
        with open(path, errors='replace') as f:
            for line in f:
                p = line.split()
                if not p: continue
                if   p[0]=='newmtl' and len(p)>1: cur=p[1]
                elif p[0]=='Kd'     and cur and len(p)>=4:
                    self._mtl_kd[cur]=[float(p[1]),float(p[2]),float(p[3])]
                elif p[0]=='map_Kd' and cur: self._mtl_tex[cur]=p[-1]

    def _load_tex(self, name):
        if not name: return 0
        for c in [name, os.path.join(self.base_dir,name),
                  os.path.join(self.base_dir,os.path.basename(name))]:
            if c in self._tex_cache: return self._tex_cache[c]
            if os.path.exists(c):
                try:
                    img=Image.open(c).convert('RGBA')
                    d=np.array(list(img.getdata()),np.uint8)
                    tid=GL.glGenTextures(1)
                    GL.glBindTexture(GL.GL_TEXTURE_2D,tid)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_LINEAR)
                    GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGBA,img.width,img.height,0,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,d)
                    GL.glBindTexture(GL.GL_TEXTURE_2D,0)
                    self._tex_cache[c]=tid; return tid
                except: pass
        return 0

    # ── Draw ──────────────────────────────────────────────────────────────────
    def _draw_g(self, shader, g, color_override):
        tint = GL.glGetUniformLocation(shader,'colorTint')
        utex = GL.glGetUniformLocation(shader,'useTexture')
        base = GL.glGetUniformLocation(shader,'baseColor')
        GL.glBindVertexArray(g['vao'])
        GL.glUniform3f(tint,1,1,1)
        if g['tex']:
            GL.glUniform1i(utex,1)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D,g['tex'])
            GL.glUniform1i(GL.glGetUniformLocation(shader,'diffuseTex'),0)
            if color_override and not g['is_glass']:
                GL.glUniform3f(tint,*color_override)
        else:
            GL.glUniform1i(utex,0)
            c = color_override if (color_override and not g['is_glass']) else g['kd']
            GL.glUniform3f(base,*c)
        GL.glDrawArrays(GL.GL_TRIANGLES,0,g['n'])
        GL.glBindVertexArray(0)

    def draw(self, shader, color_override=None, model_matrix=None, wheel_angle=0.0):
        from libs.transform import translate, rotate as rot_fn
        model_loc = GL.glGetUniformLocation(shader,'model')

        # Static parts – use whatever model matrix is already set
        for g in self._static:
            self._draw_g(shader, g, color_override)

        # Rotating wheel parts – compose car_model @ local_wheel_rot
        if model_matrix is not None:
            for g in self._rotating:
                cx,cy,cz = g['wheel_center']
                local = (translate([cx,cy,cz])
                         @ rot_fn((1,0,0), np.degrees(wheel_angle))
                         @ translate([-cx,-cy,-cz]))
                GL.glUniformMatrix4fv(model_loc,1,GL.GL_TRUE, model_matrix @ local)
                self._draw_g(shader, g, color_override)
            # Restore original model matrix
            GL.glUniformMatrix4fv(model_loc,1,GL.GL_TRUE, model_matrix)
        else:
            for g in self._rotating:
                self._draw_g(shader, g, color_override)
