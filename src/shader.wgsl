// shader.wgsl
//
// “Smokey” version:
// - Adds an animated curl-noise flow field (swirly smoke motion)
// - Adds tiny Brownian jitter (micro turbulence)
// - Adds per-particle mass + drag (so they react differently)
// - Keeps your soft-edged “explosion” that also paints a tint
// - Keeps your render pass (instanced quads) unchanged

// -------------------------
// Uniforms / storage
// -------------------------
struct Params {
    // --- simulation ---
    dt: f32,
    time: f32,                  // <--- MUST be updated from CPU (seconds)
    repel_on: u32,
    _pad0: u32,                 // align to 16

    repel_pos: vec2<f32>,       // 0..1
    repel_strength: f32,
    repel_radius: f32,
    damping: f32,

    // --- rendering ---
    point_size_px: f32,
    res: vec2<f32>,

    // --- smoke flow ---
    flow_strength: f32,         // how strongly curl flow pushes particles
    flow_scale: f32,            // noise scale (bigger = more swirls per area)

    jitter_strength: f32,       // tiny random jitter (turbulence)
    speed_color_max: f32,       // speed where tint is fully visible

    // --- painting ---
    blast_color: vec4<f32>,     // rgba (tint)
};

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    tint: vec4<f32>,
    inv_mass: f32,
    drag: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> particles_rw: array<Particle>;

@group(0) @binding(2)
var<storage, read> particles: array<Particle>;

// -------------------------
// Helpers: hash / noise / curl
// -------------------------
fn hash12(p: vec2<f32>) -> f32 {
    // Deterministic-ish pseudo-random in [0,1)
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let x = hash12(p);
    let y = hash12(p + vec2<f32>(19.19, 7.77));
    return vec2<f32>(x, y);
}

fn smooth2(t: vec2<f32>) -> vec2<f32> {
    // smoothstep-like curve for interpolation
    return t * t * (3.0 - 2.0 * t);
}

// Simple value noise (continuous, cheap)
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

// 2-octave fBm for richer swirls
fn fbm(p: vec2<f32>) -> f32 {
    var f = 0.0;
    var a = 0.6;
    var pp = p;

    f += a * value_noise(pp);
    pp = pp * 2.02 + vec2<f32>(17.0, 9.0);
    a *= 0.5;
    f += a * value_noise(pp);

    return f; // ~0..1
}

// Curl of scalar noise => divergence-free flow (swirly smoke)
fn curl_noise(p: vec2<f32>) -> vec2<f32> {
    let e = 0.35; // step in noise space; larger = smoother/softer
    let n1 = fbm(p + vec2<f32>(0.0, e));
    let n2 = fbm(p - vec2<f32>(0.0, e));
    let n3 = fbm(p + vec2<f32>(e, 0.0));
    let n4 = fbm(p - vec2<f32>(e, 0.0));

    let dndy = (n1 - n2) / (2.0 * e);
    let dndx = (n3 - n4) / (2.0 * e);

    // (dN/dy, -dN/dx)
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

    // --- Explosion / repeller (VERY soft, no hard edge) ---
    if (params.repel_on != 0u) {
        let d = p.pos - params.repel_pos;
        let dist = length(d);

        let dir0 = select(vec2<f32>(0.0, 0.0), d / max(dist, 1e-5), dist > 1e-5);

        // Gaussian-ish falloff: no boundary ring
        let x = dist / max(params.repel_radius, 1e-6);
        let w = exp(-4.5 * x * x); // softer edge, try 2.5..7.0

        // Add a bit of “dirty” direction so it’s not perfectly radial
        let n = hash12(p.pos * 37.0 + vec2<f32>(f32(i) * 0.001, params.time * 0.1));
        let ang = (n - 0.5) * 0.9; // radians, try 0.3..1.2
        let dir = rot(dir0, ang);

        // Per-particle strength jitter
        let sj = 0.75 + 0.6 * hash12(p.pos * 19.0 + vec2<f32>(0.7, f32(i) * 0.001));

        // Apply acceleration scaled by inv_mass
        let accel = params.repel_strength * w * sj * p.inv_mass;
        p.vel = p.vel + dir * (accel * params.dt);

        // Store tint (stronger in core, but still smooth)
        let tint_amt = w;
        p.tint = mix(p.tint, params.blast_color, tint_amt);
    }

    // --- Smokey flow field (curl noise) ---
    // Animate noise by shifting input with time.
    // flow_scale: bigger => more swirls per UV unit
    let flow_p = p.pos * params.flow_scale + vec2<f32>(params.time * 0.12, params.time * 0.09);

    // Curl flow is naturally swirly and “smoke-like”
    var flow = curl_noise(flow_p);

    // Mild random micro-jitter (Brownian-ish)
    // (small and scaled by dt so it doesn’t explode with framerate)
    let r = hash22(p.pos * 91.0 + vec2<f32>(f32(i) * 0.003, params.time * 0.7)) - vec2<f32>(0.5, 0.5);
    let jitter = r * (params.jitter_strength * sqrt(max(params.dt, 1e-6)));

    // Apply flow + jitter. Scale by inv_mass so heavier particles move less.
    p.vel = p.vel + (flow * params.flow_strength + jitter) * (params.dt * p.inv_mass);

    // --- Damping + per-particle drag variation ---
    // Extra drag makes particles differ and removes “perfect” motion.
    let drag = params.damping * (1.0 - clamp(p.drag, 0.0, 0.2));
    p.vel = p.vel * drag;

    // --- Integrate ---
    p.pos = p.pos + p.vel * params.dt;

    // --- Soft boundary: push back near edges (less “bouncy”, more smoky containment) ---
    // Instead of hard bounce, we apply a soft restoring force near the walls.
    // This avoids mirror-perfect reflections.
    let border = 0.04;      // thickness of the boundary zone in UV
    let wall_k = 18.0;      // strength of soft wall push

    // Left
    if (p.pos.x < border) {
        let t = (border - p.pos.x) / border;
        p.vel.x += wall_k * t * t * params.dt;
    }
    // Right
    if (p.pos.x > 1.0 - border) {
        let t = (p.pos.x - (1.0 - border)) / border;
        p.vel.x -= wall_k * t * t * params.dt;
    }
    // Bottom
    if (p.pos.y < border) {
        let t = (border - p.pos.y) / border;
        p.vel.y += wall_k * t * t * params.dt;
    }
    // Top
    if (p.pos.y > 1.0 - border) {
        let t = (p.pos.y - (1.0 - border)) / border;
        p.vel.y -= wall_k * t * t * params.dt;
    }

    // Keep inside (clamp position softly to avoid runaway)
    p.pos = clamp(p.pos, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));

    // --- Speed-tied whitening (final visible color) ---
    // Less speed => whiter. More speed => show stored tint.
    let speed = length(p.vel);
    let t_vis = clamp(speed / max(params.speed_color_max, 1e-6), 0.0, 1.0);
    let w_vis = t_vis * t_vis; // stronger whitening when slow

    let white = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    p.tint = mix(white, p.tint, w_vis);

    particles_rw[i] = p;
}

// -------------------------
// Render: instanced quads
// -------------------------
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>, // local -1..1
    @location(1) col: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) quad_pos: vec2<f32>,              // per-vertex: [-1..1] square
    @builtin(instance_index) inst: u32
) -> VSOut {
    let p = particles[inst];

    let center_clip = vec2<f32>(p.pos.x * 2.0 - 1.0, p.pos.y * 2.0 - 1.0);

    let sx = (params.point_size_px / params.res.x) * 2.0;
    let sy = (params.point_size_px / params.res.y) * 2.0;

    let clip = center_clip + vec2<f32>(quad_pos.x * sx, quad_pos.y * sy);

    var out: VSOut;
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.local_uv = quad_pos;
    out.col = p.tint;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let r = length(in.local_uv);
    let alpha = smoothstep(1.0, 0.7, r);
    return vec4<f32>(in.col.rgb, in.col.a * alpha);
}
