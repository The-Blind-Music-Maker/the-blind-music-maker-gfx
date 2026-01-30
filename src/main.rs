use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const PARTICLE_COUNT: u32 = 500_000;
const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    // 16 bytes
    dt: f32,
    time: f32,
    repel_on: u32,
    _pad0: u32,

    // 16 bytes
    repel_pos: [f32; 2],
    repel_strength: f32,
    repel_radius: f32,

    // 16 bytes
    damping: f32,
    point_size_px: f32,
    res: [f32; 2],

    // 16 bytes (smoke controls)
    flow_strength: f32,
    flow_scale: f32,
    jitter_strength: f32,
    speed_color_max: f32,

    // 16 bytes
    blast_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    tint: [f32; 4], // rename from color -> tint (recommended)
    inv_mass: f32,  // 1/mass
    drag: f32,      // extra drag, e.g. 0..0.05
    _pad: [f32; 2], // padding to 16-byte multiple (size = 48 bytes)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
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

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // Buffers
    params: Params,
    params_buf: wgpu::Buffer,
    particles_buf: wgpu::Buffer,
    quad_vb: wgpu::Buffer,

    // Bind group
    bind_group_compute: wgpu::BindGroup,
    bind_group_render: wgpu::BindGroup,

    // Pipelines
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,

    // Input
    mouse_uv: [f32; 2],
    repel_on: bool,

    last_frame: Instant,

    repel_timer: f32,
    click_color_index: u32,

    frame_count: u32,
    fps_timer: Instant,
}

impl<'a> State<'a> {
    async fn new(window: &'a Window) -> Self {
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

        // Initialize particles (UV 0..1)
        let mut rng = rand::thread_rng();
        let mut particles = Vec::with_capacity(PARTICLE_COUNT as usize);
        for _ in 0..PARTICLE_COUNT {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);
            let vx = rng.gen_range(-0.05..0.05);
            let vy = rng.gen_range(-0.05..0.05);
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

        // 16 bytes
        // dt: f32,
        // time: f32,
        // repel_on: u32,
        // _pad0: u32,

        // // 16 bytes
        // repel_pos: [f32; 2],
        // repel_strength: f32,
        // repel_radius: f32,

        // // 16 bytes
        // damping: f32,
        // point_size_px: f32,
        // res: [f32; 2],

        // // 16 bytes (smoke controls)
        // flow_strength: f32,
        // flow_scale: f32,
        // jitter_strength: f32,
        // speed_color_max: f32,

        // // 16 bytes
        // blast_color: [f32; 4],

        let params = Params {
            dt: 0.016,
            time: 0.0,
            repel_on: 0,
            _pad0: 0,

            repel_pos: [0.5, 0.5],
            repel_strength: 2.0, // try 1..10
            repel_radius: 0.25,  // try 0.05..0.4

            damping: 0.995,     // try 0.98..0.9995
            point_size_px: 2.0, // try 1..6
            res: [config.width as f32, config.height as f32],

            flow_strength: 0.2,
            flow_scale: 2.0,
            jitter_strength: 0.25,
            speed_color_max: 0.02,

            blast_color: [0.85, 0.90, 1.0, 1.0],
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_buf"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Quad vertex buffer (2 triangles, 6 vertices)
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

        // Bind group layout: params uniform + particles storage
        // Compute: params + particles (read_write)
        let bgl_compute = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_compute"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // <-- read_write
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Render: params + particles (read_only)
        let bgl_render = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_render"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2, // <-- changed from 1 to 2
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
                    binding: 2, // <-- changed from 1 to 2
                    resource: particles_buf.as_entire_binding(),
                },
            ],
        });

        // Compute pipeline

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

        // Render pipeline
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

        Self {
            surface,
            device,
            queue,
            config,
            size,

            params,
            params_buf,
            particles_buf,
            quad_vb,

            bind_group_compute,
            bind_group_render,
            compute_pipeline,
            render_pipeline,

            mouse_uv: [0.5, 0.5],
            repel_on: false,

            last_frame: Instant::now(),

            repel_timer: 0.0,
            click_color_index: 0,

            frame_count: 0,
            fps_timer: Instant::now(),
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.params.res = [self.config.width as f32, self.config.height as f32];
        self.surface.configure(&self.device, &self.config);
    }

    fn set_mouse_position(&mut self, x: f64, y: f64) {
        let w = self.config.width.max(1) as f64;
        let h = self.config.height.max(1) as f64;

        // Convert window coords (origin top-left) to UV (origin bottom-left)
        let u = (x / w).clamp(0.0, 1.0) as f32;
        let v = (1.0 - (y / h)).clamp(0.0, 1.0) as f32;

        self.mouse_uv = [u, v];
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.033);
        self.last_frame = now;

        self.repel_timer = (self.repel_timer - dt).max(0.0);
        self.repel_on = self.repel_timer > 0.0;

        self.params.dt = dt;
        self.params.repel_on = if self.repel_on { 1 } else { 0 };
        self.params.repel_pos = self.mouse_uv;
        self.params.time += dt;

        // Upload params
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    fn step_compute(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.compute_pipeline);
        // cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.set_bind_group(0, &self.bind_group_compute, &[]);

        let workgroups = (PARTICLE_COUNT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        // 1) compute step
        self.step_compute(&mut encoder);

        // 2) render step
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
                            a: 1.0,
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

            // Draw 6 vertices per instance, instance count = PARTICLE_COUNT
            rpass.draw(0..6, 0..PARTICLE_COUNT);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("GPU Particles (compute + render)")
        .with_inner_size(PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let mut state = pollster::block_on(State::new(&window));

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(sz) => state.resize(sz),

                    WindowEvent::CursorMoved { position, .. } => {
                        state.set_mouse_position(position.x, position.y);
                    }

                    WindowEvent::MouseInput {
                        state: s, button, ..
                    } => {
                        if button == MouseButton::Left && s == ElementState::Pressed {
                            state.repel_timer = 0.12; // 120ms blast

                            state.click_color_index = state.click_color_index.wrapping_add(1);
                            // simple palette (feel free to tweak)
                            state.params.blast_color = match state.click_color_index % 3 {
                                0 => [1.0, 0.0, 0.0, 1.0],
                                1 => [0.0, 1.0, 0.0, 1.0],
                                2 => [0.0, 0.0, 1.0, 1.0],
                                _ => [1.0, 1.0, 1.0, 1.0],
                            };
                        }
                    }

                    _ => {}
                },
                Event::AboutToWait => {
                    state.update();

                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(_) => {}
                    }

                    // ---- FPS COUNTER ----
                    state.frame_count += 1;
                    let elapsed = state.fps_timer.elapsed().as_secs_f32();

                    if elapsed >= 1.0 {
                        let fps = state.frame_count as f32 / elapsed;
                        println!("FPS: {:.1}", fps);
                        // window.set_title(&format!("GPU Particles — FPS: {:.1}", fps));

                        state.frame_count = 0;
                        state.fps_timer = Instant::now();
                    }
                }

                _ => {}
            }
        })
        .unwrap();
}
