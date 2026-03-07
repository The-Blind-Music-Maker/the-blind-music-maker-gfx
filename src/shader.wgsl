// shader.wgsl (MIDI + smoke repellers)

// -------------------------
// Uniforms / storage
// -------------------------
struct Params {
  dt: f32,
  time: f32,
  repel_count: u32,
  _pad0: u32,

  damping: f32,
  point_size_px: f32,
  res: vec2<f32>,

  flow_strength: f32,
  flow_scale: f32,
  jitter_strength: f32,
  speed_color_max: f32,

  score_rank: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
};



struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    tint: vec4<f32>,
    inv_mass: f32,
    drag: f32,
    _pad: vec2<f32>,
};

struct Repeller {
    pos: vec2<f32>,
    radius: f32,
    strength: f32,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> particles_rw: array<Particle>;

@group(0) @binding(2)
var<storage, read> particles: array<Particle>;

@group(0) @binding(3)
var<storage, read> repellers: array<Repeller>;

// -------------------------
// Helpers: hash / noise / curl
// -------------------------
fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let x = hash12(p);
    let y = hash12(p + vec2<f32>(19.19, 7.77));
    return vec2<f32>(x, y);
}

fn smooth2(t: vec2<f32>) -> vec2<f32> {
    return t * t * (3.0 - 2.0 * t);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = smooth2(f);

    let a = hash12(i + vec2<f32>(0.0, 0.0));
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));

    let x1 = mix(a, b, u.x);
    let x2 = mix(c, d, u.x);
    return mix(x1, x2, u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var f = 0.0;
    var a = 0.6;
    var pp = p;

    f += a * value_noise(pp);
    pp = pp * 2.02 + vec2<f32>(17.0, 9.0);
    a *= 0.5;
    f += a * value_noise(pp);

    return f;
}

fn curl_noise(p: vec2<f32>) -> vec2<f32> {
    let e = 0.35;
    let n1 = fbm(p + vec2<f32>(0.0, e));
    let n2 = fbm(p - vec2<f32>(0.0, e));
    let n3 = fbm(p + vec2<f32>(e, 0.0));
    let n4 = fbm(p - vec2<f32>(e, 0.0));

    let dndy = (n1 - n2) / (2.0 * e);
    let dndx = (n3 - n4) / (2.0 * e);

    return vec2<f32>(dndy, -dndx);
}

fn rot(v: vec2<f32>, ang: f32) -> vec2<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}

// -------------------------
// Compute: simulate + paint + smoke flow
// -------------------------
@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&particles_rw)) { return; }

    var p = particles_rw[i];

    // --- Repellers (MIDI notes + mouse) ---
    for (var k: u32 = 0u; k < params.repel_count; k = k + 1u) {
        let r = repellers[k];

        let d = p.pos - r.pos;
        let dist = length(d);

        let dir0 = select(vec2<f32>(0.0, 0.0), d / max(dist, 1e-5), dist > 1e-5);

        // Gaussian falloff: no boundary ring
        let x = dist / max(r.radius, 1e-6);
        let w = exp(-4.5 * x * x);

        if (w > 0.00005) {
            // A bit of “dirty” direction so it’s not perfectly radial
            let n = hash12(p.pos * 37.0 + vec2<f32>(f32(i) * 0.001 + f32(k), params.time * 0.1));
            let ang = (n - 0.5) * 0.7; // radians
            let dir = rot(dir0, ang);

            // Per-particle strength jitter
            let sj = 0.75 + 0.6 * hash12(p.pos * 19.0 + vec2<f32>(0.7 + f32(k), f32(i) * 0.001));

            // Strength already contains MIDI velocity scaling (CPU side)
            let accel = r.strength * w * sj * p.inv_mass;
            p.vel = p.vel + dir * (accel * params.dt);

            // Tint toward repeller color
            p.tint = mix(p.tint, r.color, w);
        }
    }

    // --- Smokey flow field (curl noise) ---
    let flow_p = p.pos * params.flow_scale + vec2<f32>(params.time * 0.12, params.time * 0.09);
    let flow = curl_noise(flow_p);

    let rj = hash22(p.pos * 91.0 + vec2<f32>(f32(i) * 0.003, params.time * 0.7)) - vec2<f32>(0.5, 0.5);
    let jitter = rj * (params.jitter_strength * sqrt(max(params.dt, 1e-6)));

    p.vel = p.vel + (flow * params.flow_strength + jitter) * (params.dt * p.inv_mass);

    // --- Damping + per-particle drag ---
    let drag = params.damping * (1.0 - clamp(p.drag, 0.0, 0.2));
    p.vel = p.vel * drag;

    // --- Integrate ---
    p.pos = p.pos + p.vel * params.dt;

    // --- Soft boundary containment ---
    let border = 0.04;
    let wall_k = 18.0;

    if (p.pos.x < border) {
        let t = (border - p.pos.x) / border;
        p.vel.x += wall_k * t * t * params.dt;
    }
    if (p.pos.x > 1.0 - border) {
        let t = (p.pos.x - (1.0 - border)) / border;
        p.vel.x -= wall_k * t * t * params.dt;
    }
    if (p.pos.y < border) {
        let t = (border - p.pos.y) / border;
        p.vel.y += wall_k * t * t * params.dt;
    }
    if (p.pos.y > 1.0 - border) {
        let t = (p.pos.y - (1.0 - border)) / border;
        p.vel.y -= wall_k * t * t * params.dt;
    }

    p.pos = clamp(p.pos, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));

    // --- Speed-tied whitening ---
    let speed = length(p.vel);
    let t_vis = clamp(speed / max(params.speed_color_max, 1e-6), 0.0, 1.0);
    let w_vis = t_vis * t_vis;

    let white = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    p.tint = mix(white, p.tint, w_vis);

    particles_rw[i] = p;
}

// -------------------------
// Render: instanced quads
// -------------------------
// -------------------------
// Vertex output
// -------------------------
struct VSOut {
    @builtin(position) pos: vec4<f32>,  // final clip-space position
    @location(0) local_uv: vec2<f32>,   // quad coordinates (-1..1)
    @location(1) col: vec4<f32>,        // particle color
    @location(2) screen_uv: vec2<f32>,  // screen coordinates (0..1)
};

// -------------------------
// Vertex shader
// -------------------------
@vertex
fn vs_main(
    @location(0) quad_pos: vec2<f32>,
    @builtin(instance_index) inst: u32
) -> VSOut {

    // particle data
    let p = particles[inst];

    // particle center in clip space
    let center_clip = vec2<f32>(
        p.pos.x * 2.0 - 1.0,
        p.pos.y * 2.0 - 1.0
    );

    // convert point size (pixels) → clip-space size
    let sx = (params.point_size_px / params.res.x) * 2.0;
    let sy = (params.point_size_px / params.res.y) * 2.0;

    // quad vertex offset
    let clip_xy = center_clip + vec2<f32>(
        quad_pos.x * sx,
        quad_pos.y * sy
    );

    var out: VSOut;

    // final position
    out.pos = vec4<f32>(clip_xy, 0.0, 1.0);

    // used to compute particle circle
    out.local_uv = quad_pos;

    // particle color
    out.col = p.tint;

    // convert clip space [-1..1] → screen UV [0..1]
    out.screen_uv = clip_xy * 0.5 + vec2<f32>(0.5, 0.5);

    return out;
}

fn rank_u32(score_rank: f32) -> u32 {
    // if CPU sends exact 0/1/2/3 in score_rank, this works.
    // rounding avoids issues if it's slightly off.
    return min(u32(score_rank + 0.5), 3u);
}

fn srgb_channel_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb8_to_linear(r: f32, g: f32, b: f32) -> vec3<f32> {
    let srgb = vec3<f32>(r, g, b) / 255.0;
    return vec3<f32>(
        srgb_channel_to_linear(srgb.r),
        srgb_channel_to_linear(srgb.g),
        srgb_channel_to_linear(srgb.b)
    );
}

fn rank_color(rank: u32) -> vec3<f32> {
    if (rank == 0u) { return srgb8_to_linear(231.0, 83.0, 66.0); }   // #E75342
    if (rank == 1u) { return srgb8_to_linear(233.0, 84.0, 66.0); }   // #E95442
    if (rank == 2u) { return srgb8_to_linear(253.0, 195.0, 76.0); }  // #FDC34C
    return srgb8_to_linear(72.0, 184.0, 101.0);                      // #48B865
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // your existing particle shape
    let r = length(in.local_uv);
    let particle_alpha = smoothstep(1.0, 0.7, r);
    var col = vec4<f32>(in.col.rgb, in.col.a * particle_alpha);

    // HUD rectangle overlay (bottom-right)
    let uv = in.screen_uv;
    let rank = rank_u32(params.score_rank);
    let hud = rank_color(rank);

    // Rectangle bounds in screen UV
    // bottom-right corner, with a little margin
    let margin = vec2<f32>(0.02, 0.02);
    let size   = vec2<f32>(0.10, 0.06); // width, height (tweak these)
    let minp   = vec2<f32>(1.0, 0.0) - margin - size;
    let maxp   = vec2<f32>(1.0, 0.0) - margin;

    let inside =
        uv.x >= minp.x && uv.x <= maxp.x &&
        uv.y >= minp.y && uv.y <= maxp.y;

    if (inside) {
        // blend a solid rectangle on top
        col.r = hud.x;
        col.g = hud.y;
        col.b = hud.z;
        col.a = 1.0;
    }

    return col;
}

// -------------------------
// HUD full-screen pass
// -------------------------
struct HudVSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_hud(@location(0) quad_pos: vec2<f32>) -> HudVSOut {
    var out: HudVSOut;

    // fullscreen quad already in clip space
    out.pos = vec4<f32>(quad_pos, 0.0, 1.0);

    // convert [-1..1] → [0..1]
    out.uv = quad_pos * 0.5 + vec2<f32>(0.5, 0.5);

    return out;
}

@fragment
fn fs_hud(in: HudVSOut) -> @location(0) vec4<f32> {
    let uv = in.uv;

    let rank = rank_u32(params.score_rank);
    let color = rank_color(rank);

    // square size in pixels
    let square_px = 60.0;

    // convert to UV
    let size = vec2<f32>(
        square_px / params.res.x,
        square_px / params.res.y
    );

    // margin from screen edge (in pixels)
    let margin_px = 20.0;

    let margin = vec2<f32>(
        margin_px / params.res.x,
        margin_px / params.res.y
    );

    // bottom-right square
    let minp = vec2<f32>(1.0 - margin.x - size.x, margin.y);
    let maxp = vec2<f32>(1.0 - margin.x,          margin.y + size.y);

    let inside =
        uv.x >= minp.x && uv.x <= maxp.x &&
        uv.y >= minp.y && uv.y <= maxp.y;

    if (inside) {
        return vec4<f32>(color, 1.0);
    }

    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}