// main.rs
//
// GPU particles + smoke flow + MIDI grid repellers (via channel)
// - Compute shader reads an array of Repeller structs (@binding(3))
// - Each frame we build repellers from currently-held MIDI notes + optional mouse blast
//
// NOTE: This file assumes your shader.wgsl is the "MIDI + smoke repellers" version
//       I sent earlier (with Params including repel_count, and repellers buffer at binding(3)).

use std::{sync::mpsc::Receiver, time::Instant};

use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod midi;
mod state;

use crate::{
    midi::{MidiMsg, spawn_midi_listener},
    state::State,
};

fn main() {
    // Hook point: provide a Receiver<MidiMsg> here when you have MIDI set up.
    // Example:
    let (tx, rx) = std::sync::mpsc::channel::<MidiMsg>();
    spawn_midi_listener(tx); // your code

    let midi_rx_option: Option<Receiver<MidiMsg>> = Some(rx);

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("GPU Particles (smoke + MIDI repellers)")
        .with_inner_size(PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let mut state = pollster::block_on(State::new(&window, midi_rx_option));

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
                            state.repel_timer = 0.12; // 120ms mouse blast
                            state.click_color_index = state.click_color_index.wrapping_add(1);
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

                    // FPS counter (prints once per ~second)
                    state.frame_count += 1;
                    let elapsed = state.fps_timer.elapsed().as_secs_f32();
                    if elapsed >= 1.0 {
                        let fps = state.frame_count as f32 / elapsed;
                        // println!("FPS: {:.1} | repellers: {}", fps, state.params.repel_count);
                        state.frame_count = 0;
                        state.fps_timer = Instant::now();
                    }
                }

                _ => {}
            }
        })
        .unwrap();
}
