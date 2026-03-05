use std::{
    error::Error,
    io::{Write, stdin, stdout},
    sync::mpsc,
    thread,
    time::Duration,
};

use midir::{Ignore, MidiInput, MidiInputPort};

// -------------------------
// MIDI message type (you provide a channel that sends these)
// -------------------------
#[derive(Debug, Clone, Copy)]
pub enum MidiMsg {
    NoteOn { note: u8, vel: u8, chan: u8 }, // vel 1..127
    NoteOff { note: u8, chan: u8 },
    Cc { controller: u8, value: u8, chan: u8 },
}

fn open_midi_input() -> Result<(MidiInput, MidiInputPort), Box<dyn Error>> {
    let mut midi_in = MidiInput::new("midir-cc-listener")?;
    midi_in.ignore(Ignore::None);

    let in_ports = midi_in.ports();
    let in_port: MidiInputPort = match in_ports.len() {
        0 => return Err("no input port found".into()),
        1 => {
            println!(
                "Choosing the only available input port: {}",
                midi_in.port_name(&in_ports[0]).unwrap()
            );
            in_ports[0].clone()
        }
        _ => {
            println!("\nAvailable input ports:");
            for (i, p) in in_ports.iter().enumerate() {
                println!("{}: {}", i, midi_in.port_name(p).unwrap());
            }
            print!("Please select input port: ");
            stdout().flush()?;
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            in_ports
                .get(input.trim().parse::<usize>()?)
                .ok_or("invalid input port selected")?
                .clone()
        }
    };

    Ok((midi_in, in_port))
}

pub fn spawn_midi_listener(midi_tx: mpsc::Sender<MidiMsg>) -> thread::JoinHandle<()> {
    let Some(midi_data): Option<(MidiInput, MidiInputPort)> = (match open_midi_input() {
        Ok(x) => Some(x),
        Err(e) => {
            eprintln!("MIDI input open failed: {e}");
            None
        }
    }) else {
        todo!()
    };

    let midi_in = midi_data.0;
    let in_port = midi_data.1;

    thread::spawn(move || {
        let conn = midi_in.connect(
            &in_port,
            "midir-listener",
            move |_stamp, message, _| {
                if message.len() < 3 {
                    return;
                }

                let status = message[0];
                let data1 = message[1];
                let data2 = message[2];

                // Ignore realtime/other system messages (status >= 0xF0)
                if status >= 0xF0 {
                    return;
                }

                let kind = status & 0xF0;
                let ch = (status & 0x0F) + 1; // 1..16

                if ch == 5 && kind == 0xB0 {
                    let _ = midi_tx.send(MidiMsg::Cc {
                        controller: data1,
                        value: data2,
                        chan: ch,
                    });
                    return;
                }

                if ch > 3 {
                    return;
                }

                match kind {
                    0x90 => {
                        // println!("chan: {ch}, note: {data1}");
                        // Note On, but vel=0 is Note Off
                        if data2 == 0 {
                            let _ = midi_tx.send(MidiMsg::NoteOff {
                                note: data1,
                                chan: ch,
                            });
                        } else {
                            let _ = midi_tx.send(MidiMsg::NoteOn {
                                note: data1,
                                vel: data2,
                                chan: ch,
                            });
                        }
                    }
                    0x80 => {
                        let _ = midi_tx.send(MidiMsg::NoteOff {
                            note: data1,
                            chan: ch,
                        });
                    }
                    _ => {} // ignore other messages for now
                }
            },
            (),
        );

        let _conn = match conn {
            Ok(c) => c,
            Err(e) => {
                eprintln!("MIDI input connect failed: {e}");
                return;
            }
        };

        loop {
            thread::sleep(Duration::from_secs(3600));
        }
    })
}
