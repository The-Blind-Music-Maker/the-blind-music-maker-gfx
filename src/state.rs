use std::{collections::HashMap, sync::mpsc::Receiver, time::Instant};

use crate::midi::MidiMsg;
use bytemuck::{Pod, Zeroable};
use rand::Rng;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

const PARTICLE_COUNT: u32 = 500_000;
const WORKGROUP_SIZE: u32 = 256;

// MIDI mapping
const NOTE_MIN: u8 = 24;
const NOTE_MAX: u8 = 108;
const GRID_COLS: u32 = 9;
const GRID_ROWS: u32 = 12;

const MAX_REPELLERS: usize = 64;

// -------------------------
// Helper mapping functions
// -------------------------
// fn note_to_uv(note: u8) -> [f32; 2] {
//     let idx = (note - NOTE_MIN) as u32; // 0..86
//     let col = idx % GRID_COLS;
//     let row = idx / GRID_COLS; // 0..7

//     let u = (col as f32 + 0.5) / (GRID_COLS as f32);
//     let v = (row as f32 + 0.5) / (GRID_ROWS as f32);
//     [u, v]
// }

fn note_to_uv(note: u8) -> [f32; 2] {
    let n = note.clamp(NOTE_MIN, NOTE_MAX);
    let idx = (n - NOTE_MIN) as u32;

    let col = idx % GRID_COLS;
    let row = idx / GRID_COLS;

    let max_idx = (NOTE_MAX - NOTE_MIN) as u32;
    let max_row = max_idx / GRID_COLS; // last row that is actually used

    // map row 0..max_row -> 0..(GRID_ROWS-1)
    let row_scaled = if max_row > 0 {
        (row as f32 / max_row as f32) * (GRID_ROWS as f32 - 1.0)
    } else {
        0.0
    };

    let u = (col as f32 + 0.5) / GRID_COLS as f32;
    let v = (row_scaled + 0.5) / GRID_ROWS as f32;

    [u, v]
}

// Velocity -> strength (nonlinear feels more musical)
fn velocity_to_strength(vel: u8) -> f32 {
    let v01 = vel as f32 / 127.0;
    let v = v01 * v01; // quadratic curve
    let base = 1.5;
    let range = 22.0;
    base + range * v * 0.07
}

fn note_to_radius(note: u8, vel: u8) -> f32 {
    // clamp to your audible range
    let n = note.clamp(NOTE_MIN, NOTE_MAX);

    // 0 at low notes, 1 at high notes
    let t = (n - NOTE_MIN) as f32 / (NOTE_MAX - NOTE_MIN) as f32;

    // invert so high notes -> 0
    let inv = 1.0 - t;

    // choose min/max radius (UV units)
    let r_min = 0.04; // smallest for highest notes
    let r_max = 0.15; // biggest for lowest notes

    // optional: still let velocity slightly affect radius
    let v01 = vel as f32 / 127.0;
    let vel_boost = 0.6 + 0.4 * v01; // 0.6..1.0

    (r_min + (r_max - r_min) * inv) * vel_boost
}

fn note_to_color(note: u8, chan: u8) -> [f32; 4] {
    // simple pitch-class palette
    match chan % 12 {
        0 => [1.0, 0.2, 0.2, 1.0],
        9 => [1.0, 0.5, 0.2, 1.0],
        2 => [1.0, 0.8, 0.2, 1.0],
        3 => [0.2, 1.0, 0.8, 1.0],
        4 => [0.6, 1.0, 0.2, 1.0],
        5 => [0.2, 1.0, 0.3, 1.0],
        6 => [0.2, 0.7, 1.0, 1.0],
        7 => [0.2, 0.3, 1.0, 1.0],
        8 => [0.6, 0.2, 1.0, 1.0],
        1 => [1.0, 0.2, 1.0, 1.0],
        10 => [1.0, 0.2, 0.6, 1.0],
        _ => [0.9, 0.9, 0.9, 1.0],
    }
}

// -------------------------
// GPU structs (MUST match shader.wgsl)
// -------------------------
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    // 16 bytes
    dt: f32,
    time: f32,
    pub repel_count: u32,
    _pad0: u32,

    // 16 bytes
    damping: f32,
    point_size_px: f32,
    res: [f32; 2],

    // 16 bytes
    flow_strength: f32,
    flow_scale: f32,
    jitter_strength: f32,
    speed_color_max: f32,

    // 16 bytes (rank + padding)
    score_rank: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    tint: [f32; 4],
    inv_mass: f32,
    drag: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Repeller {
    pos: [f32; 2],
    radius: f32,
    strength: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QuadVertex {
    pos: [f32; 2], // [-1..1]
}

impl QuadVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                shader_location: 0,
                offset: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

pub struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,

    // Buffers
    pub params: Params,
    params_buf: wgpu::Buffer,
    repellers_buf: wgpu::Buffer,
    quad_vb: wgpu::Buffer,

    // Bind groups
    bind_group_compute: wgpu::BindGroup,
    bind_group_render: wgpu::BindGroup,
    bind_group_hud: wgpu::BindGroup,

    // Pipelines
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    hud_pipeline: wgpu::RenderPipeline,

    // Input (mouse as optional extra repeller)
    mouse_uv: [f32; 2],
    pub repel_timer: f32,

    // MIDI state
    midi_rx: Option<Receiver<MidiMsg>>,
    active_notes: HashMap<u8, (u8, u8)>, // note -> velocity
    repellers: Vec<Repeller>,

    // timing
    last_frame: Instant,

    // optional mouse color cycling
    pub click_color_index: u32,

    // fps counter
    pub frame_count: u32,
    pub fps_timer: Instant,

    score_rank: u8,
}

impl<'a> State<'a> {
    pub async fn new(window: &'a Window, midi_rx: Option<Receiver<MidiMsg>>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).expect("create_surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("request_adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("request_device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // -------------------------
        // Particles init
        // -------------------------
        let mut rng = rand::thread_rng();
        let mut particles = Vec::with_capacity(PARTICLE_COUNT as usize);
        for _ in 0..PARTICLE_COUNT {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);
            let vx = rng.gen_range(-0.02..0.02);
            let vy = rng.gen_range(-0.02..0.02);

            let m = rng.gen_range(0.5..2.0);
            let inv_mass = 1.0 / m;
            let drag = rng.gen_range(0.0..0.04);

            particles.push(Particle {
                pos: [x, y],
                vel: [vx, vy],
                tint: [1.0, 1.0, 1.0, 1.0],
                inv_mass,
                drag,
                _pad: [0.0, 0.0],
            });
        }

        let particles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles_buf"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // -------------------------
        // Params
        // -------------------------
        let params = Params {
            dt: 0.016,
            time: 0.0,
            repel_count: 0,
            _pad0: 0,

            damping: 0.995,
            point_size_px: 2.0,
            res: [config.width as f32, config.height as f32],

            flow_strength: 0.20,
            flow_scale: 2.0,
            jitter_strength: 0.25,
            speed_color_max: 0.02,

            score_rank: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_buf"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // -------------------------
        // Repellers buffer (storage, updated every frame)
        // -------------------------
        let repellers_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repellers_buf"),
            size: (std::mem::size_of::<Repeller>() * MAX_REPELLERS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // -------------------------
        // Quad VB (2 triangles)
        // -------------------------
        let quad = [
            QuadVertex { pos: [-1.0, -1.0] },
            QuadVertex { pos: [1.0, -1.0] },
            QuadVertex { pos: [1.0, 1.0] },
            QuadVertex { pos: [-1.0, -1.0] },
            QuadVertex { pos: [1.0, 1.0] },
            QuadVertex { pos: [-1.0, 1.0] },
        ];

        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad_vb"),
            contents: bytemuck::cast_slice(&quad),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // -------------------------
        // Bind group layouts
        // -------------------------
        // Compute: params (0), particles_rw (1), repellers (3)
        let bgl_compute = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_compute"),
            entries: &[
                // params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // particles_rw
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // repellers (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Render: params (0), particles (2)
        let bgl_render = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_render"),
            entries: &[
                // params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // particles read-only
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group_compute = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_compute"),
            layout: &bgl_compute,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: repellers_buf.as_entire_binding(),
                },
            ],
        });

        let bind_group_render = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_render"),
            layout: &bgl_render,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: particles_buf.as_entire_binding(),
                },
            ],
        });

        // -------------------------
        // Pipelines
        // -------------------------
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pl"),
                bind_group_layouts: &[&bgl_compute],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
            compilation_options: Default::default(),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pl"),
                bind_group_layouts: &[&bgl_render],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[QuadVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let bgl_hud = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_hud"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group_hud = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_hud"),
            layout: &bgl_hud,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });

        let hud_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud_pl"),
            bind_group_layouts: &[&bgl_hud],
            push_constant_ranges: &[],
        });

        let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud_pipeline"),
            layout: Some(&hud_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_hud",
                buffers: &[QuadVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_hud",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,

            params,
            params_buf,
            repellers_buf,
            quad_vb,

            bind_group_compute,
            bind_group_render,
            bind_group_hud,

            compute_pipeline,
            render_pipeline,
            hud_pipeline,

            mouse_uv: [0.5, 0.5],
            repel_timer: 0.0,

            midi_rx,
            active_notes: HashMap::new(),
            repellers: Vec::with_capacity(MAX_REPELLERS),

            last_frame: Instant::now(),
            click_color_index: 0,

            frame_count: 0,
            fps_timer: Instant::now(),
            score_rank: 0,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.params.res = [self.config.width as f32, self.config.height as f32];
        self.surface.configure(&self.device, &self.config);
    }

    pub fn set_mouse_position(&mut self, x: f64, y: f64) {
        let w = self.config.width.max(1) as f64;
        let h = self.config.height.max(1) as f64;

        // window coords -> UV, origin bottom-left
        let u = (x / w).clamp(0.0, 1.0) as f32;
        let v = (1.0 - (y / h)).clamp(0.0, 1.0) as f32;
        self.mouse_uv = [u, v];
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.033);
        self.last_frame = now;

        self.params.dt = dt;
        self.params.time += dt;
        self.params.score_rank = self.score_rank.min(3) as f32; // 0..3

        // countdown mouse blast
        self.repel_timer = (self.repel_timer - dt).max(0.0);

        // Drain MIDI channel (non-blocking)
        if let Some(rx) = self.midi_rx.as_ref() {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    MidiMsg::NoteOn { note, vel, chan } => {
                        if (NOTE_MIN..=NOTE_MAX).contains(&note) && vel != 0 {
                            self.active_notes.insert(note, (vel, chan));
                        }
                    }
                    MidiMsg::NoteOff { note, .. } => {
                        self.active_notes.remove(&note);
                    }
                    MidiMsg::Cc {
                        controller, value, ..
                    } => {
                        if controller == 16 {
                            self.score_rank = value;
                        }
                    }
                }
            }
        }

        // Build repellers list
        self.repellers.clear();

        // MIDI repellers
        for (&note, (vel, chan)) in self.active_notes.iter() {
            if self.repellers.len() >= MAX_REPELLERS {
                break;
            }

            let pos = note_to_uv(note);
            let strength = velocity_to_strength(*vel);
            // let v01 = vel as f32 / 127.0;

            // velocity also affects radius (feels good)
            // let radius = 0.10 + 0.14 * v01;
            let radius = note_to_radius(note, *vel);
            let color = note_to_color(note, *chan);

            self.repellers.push(Repeller {
                pos,
                radius,
                strength,
                color,
            });
        }

        // Optional mouse repeller (adds one more repeller while timer active)
        if self.repel_timer > 0.0 && self.repellers.len() < MAX_REPELLERS {
            // cycle mouse color a bit
            let color = match self.click_color_index % 3 {
                0 => [1.0, 0.9, 0.9, 1.0],
                1 => [0.9, 1.0, 0.9, 1.0],
                _ => [0.9, 0.9, 1.0, 1.0],
            };

            self.repellers.push(Repeller {
                pos: self.mouse_uv,
                radius: 0.22,
                strength: 10.0,
                color,
            });
        }

        // Upload repellers + params
        self.params.repel_count = self.repellers.len() as u32;
        self.queue.write_buffer(
            &self.repellers_buf,
            0,
            bytemuck::cast_slice(&self.repellers),
        );

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn step_compute(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group_compute, &[]);

        let workgroups = (PARTICLE_COUNT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        // compute
        self.step_compute(&mut encoder);

        // render
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.015,
                            a: 0.5,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bind_group_render, &[]);
            rpass.set_vertex_buffer(0, self.quad_vb.slice(..));
            rpass.draw(0..6, 0..PARTICLE_COUNT);
        }

        // HUD pass (draw fullscreen quad once, overlay)
        {
            let mut hud_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hud_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // IMPORTANT: keep particle result
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            hud_pass.set_pipeline(&self.hud_pipeline);
            hud_pass.set_bind_group(0, &self.bind_group_hud, &[]);
            hud_pass.set_vertex_buffer(0, self.quad_vb.slice(..));
            hud_pass.draw(0..6, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}
