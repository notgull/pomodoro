// MIT/Apache2 License

use breadx::{
    auto::xproto::ExposeEvent, rgb, AsyncDisplayConnection, Event, EventMask, GcParameters,
    Gcontext, Image, ImageFormat, Pixmap, Visualtype,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures_lite::{future, FutureExt, StreamExt};
use rusttype::{point, Font, Scale};
use smol::{
    channel::{unbounded, Receiver, Sender},
    lock::RwLock,
    Executor, Timer,
};
use std::{boxed::Box, iter, mem, time::Duration};

static FONT: RwLock<Option<Font<'static>>> = RwLock::new(None);

const WORK_LEN: usize = 25;
const BREAK_LEN: usize = 5;
const LBREAK_LEN: usize = 20;
const NUM_BREAKS_BEFORE_LONG_BREAK: u8 = 3;

/// Pomodoro time period type.
#[derive(Debug, Clone, Copy)]
enum TimePeriod {
    Work,
    Break,
}

/// Pomodoro iterator.
#[derive(Debug, Clone, Default)]
struct PomodoroIterator {
    num_breaks: u8,
    was_last_period_break: bool,
}

impl Iterator for PomodoroIterator {
    type Item = (usize, TimePeriod);

    #[inline]
    fn next(&mut self) -> Option<(usize, TimePeriod)> {
        if self.was_last_period_break {
            self.was_last_period_break = false;
            Some((WORK_LEN, TimePeriod::Work))
        } else {
            self.was_last_period_break = true;
            self.num_breaks += 1;
            if self.num_breaks >= NUM_BREAKS_BEFORE_LONG_BREAK {
                self.num_breaks = 0;
                Some((LBREAK_LEN, TimePeriod::Break))
            } else {
                Some((BREAK_LEN, TimePeriod::Break))
            }
        }
    }
}

/// Play a sound signifiying the change in time period.
/// Code inspired by https://github.com/RustAudio/cpal/blob/master/examples/beep.rs
/// Shouldn't block.
fn play_signifier<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> anyhow::Result<()>
where
    T: cpal::Sample,
{
    #[inline]
    fn write_data<T, F>(output: &mut [T], channels: usize, next_sample: &mut F)
    where
        T: cpal::Sample,
        F: FnMut() -> f32,
    {
        for frame in output.chunks_mut(channels) {
            let value: T = cpal::Sample::from::<f32>(&next_sample());
            for sample in frame.iter_mut() {
                *sample = value;
            }
        }
    }

    let sample_rate = config.sample_rate.0 as f32;
    let channels = config.channels as usize;

    // Write a sine wave
    let mut sample_clock = 0f32;
    let mut sampler = move || {
        sample_clock = (sample_clock + 1.0) % sample_rate;
        (sample_clock * 444.0 * 2.0 * std::f32::consts::PI / sample_rate).sin()
    };

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut sampler)
        },
        |err| eprintln!("An error occurred: {}", err),
    )?;
    stream.play()?;
    std::thread::sleep(Duration::from_millis(1000));
    Ok(())
}

/// Given that we have "x" seconds left, write this statistic to a given pixmap.
async fn draw_clock(
    conn: &mut AsyncDisplayConnection,
    font: &Font<'static>,
    seconds: usize,
    pixmap: Pixmap,
    gc: Gcontext,
    width: u16,
    height: u16,
    visual: &Visualtype,
    depth: u8,
    r: u8,
    g: u8,
    b: u8,
) -> anyhow::Result<()> {
    // the text we are rendering
    let text = format!("{}:{:02}", seconds / 60, seconds % 60);

    // we'll use a uniform font scaling
    let scale = Scale::uniform(64.0 as f32);

    // layout the glyphs in a line
    let glyphs: Vec<_> = font.layout(&text, scale, point(5.0, 5.0)).collect();

    // create an image to draw our text into
    let mut img = Image::new(
        conn,
        Some(visual),
        depth,
        ImageFormat::ZPixmap,
        0,
        iter::repeat(0)
            .take((width as usize * height as usize) * 4)
            .collect::<Box<[u8]>>(),
        width as _,
        height as _,
        32,
        None,
    )
    .unwrap();

    for glyph in glyphs {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {
                let r = (r as f32 * v) as u8;
                let g = (g as f32 * v) as u8;
                let b = (b as f32 * v) as u8;
                let color = rgb(r, g, b);
                img.set_pixel(
                    x as usize + bounding_box.min.x as usize,
                    y as usize + bounding_box.max.y as usize,
                    color,
                );
            });
        }
    }

    // put the image on the pixmap
    conn.put_image_async(pixmap, gc, &img, 0, 0, 0, 0, width as _, height as _)
        .await?;
    Ok(())
}

/// Coroutine no. 1: Connect to the X server and await messages from the timer thread.
async fn x_process(receiver: Receiver<(usize, [u8; 3])>) -> anyhow::Result<()> {
    enum MainLoopDirective {
        Event(breadx::Result<Event>),
        Channel(usize, [u8; 3]),
    }

    let mut width = 160;
    let mut height = 66;

    // create a connection, a window, and an appropriately sized pixmap
    let mut conn = AsyncDisplayConnection::create_async(None, None).await?;
    let win = conn
        .create_simple_window_async(
            conn.default_screen().root,
            0,
            0,
            width,
            height,
            0,
            conn.default_black_pixel(),
            conn.default_black_pixel(),
        )
        .await?;
    let vis = conn.default_visual().clone();
    let depth = win.geometry_immediate_async(&mut conn).await?.depth;
    let pixmap = conn.create_pixmap_async(win, width, height, depth).await?;
    let gc = conn
        .create_gc_async(
            pixmap,
            GcParameters {
                foreground: Some(conn.default_black_pixel()),
                graphics_exposures: Some(0),
                ..Default::default()
            },
        )
        .await?;

    win.set_event_mask_async(&mut conn, EventMask::EXPOSURE | EventMask::STRUCTURE_NOTIFY)
        .await?;
    win.map_async(&mut conn).await?;
    win.set_title_async(&mut conn, "seagulldoro").await?;

    let wdw = conn
        .intern_atom_immediate_async("WM_DELETE_WINDOW".to_owned(), false)
        .await?;
    win.set_wm_protocols_async(&mut conn, &[wdw]).await?;

    // begin our main loop
    loop {
        let directive = async {
            let res = receiver
                .recv()
                .await
                .expect("Sender shouldn't be dropped before receiver");
            MainLoopDirective::Channel(res.0, res.1)
        }
        .or(async { MainLoopDirective::Event(conn.wait_for_event_async().await) })
        .await;

        match directive {
            MainLoopDirective::Channel(seconds, [r, g, b]) => {
                if let Some(font) = FONT.read().await.as_ref() {
                    // reset the image
                    draw_clock(
                        &mut conn, font, seconds, pixmap, gc, 160, 66, &vis, depth, r, g, b,
                    )
                    .await?;

                    // send an exposure event to force a redraw
                    conn.send_event_async(
                        win,
                        EventMask::EXPOSURE,
                        Event::Expose(ExposeEvent {
                            window: win,
                            x: 0,
                            y: 0,
                            width,
                            height,
                            count: 0,
                            ..Default::default()
                        }),
                    )
                    .await?;
                }
            }
            MainLoopDirective::Event(ev) => match ev? {
                Event::ClientMessage(cme) => {
                    if cme.data.longs()[0] == wdw.xid {
                        break;
                    }
                }
                Event::ConfigureNotify(cne) => {
                    // update width and height
                    width = cne.width;
                    height = cne.height;
                }
                Event::Expose(_) => {
                    conn.copy_area_async(pixmap, win, gc, 0, 0, width as _, height as _, 0, 0)
                        .await?;
                }
                _ => (),
            },
        }
    }

    Ok(())
}

/// Coroutine no. 2: wait 1 second, then send an interprocess message that holds the number of
///                  seconds left in the internal timer and the associated color.
async fn timer<T>(
    sender: Sender<(usize, [u8; 3])>,
    device: cpal::Device,
    config: cpal::StreamConfig,
) -> anyhow::Result<()>
where
    T: cpal::Sample,
{
    let (mut r, mut g, mut b) = (255, 255, 255);
    let mut pomodoro = PomodoroIterator::default();
    let mut timer = Timer::interval(Duration::from_secs(1));
    let mut seconds_left = (WORK_LEN * 60) + 1;

    let mut device = Some(device);
    let mut config = Some(config);

    loop {
        seconds_left = match seconds_left.checked_sub(1) {
            Some(sl) => sl,
            None => {
                let (d, c) = (device.take().unwrap(), config.take().unwrap());
                let (d, c, res) = smol::unblock(move || {
                    let res = play_signifier::<T>(&d, &c);
                    (d, c, res)
                })
                .await;
                res?;
                device = Some(d);
                config = Some(c);

                let (ml, method) = pomodoro.next().unwrap();
                match method {
                    TimePeriod::Work => {
                        r = 255;
                        g = 255;
                        b = 255;
                    }
                    TimePeriod::Break => {
                        r = 60;
                        g = 255;
                        b = 60;
                    }
                }
                ml * 60
            }
        };

        let res = sender.send((seconds_left, [r, g, b])).await;
        if res.is_err() {
            break;
        }

        timer.next().await.unwrap();
    }

    Ok(())
}

/// Coroutine no. 3: load the font from the file.
async fn load_font() -> anyhow::Result<()> {
    let font_data = include_bytes!("font.ttf");
    let f = Font::try_from_bytes(font_data as &[u8])
        .ok_or(breadx::BreadError::StaticMsg("Failed to load font"))?;

    *FONT.write().await = Some(f);
    Ok(())
}

/// Async entry point.
async fn entry<'a>(
    ex: &Executor<'a>,
    device: cpal::Device,
    format: cpal::SampleFormat,
    config: cpal::StreamConfig,
) -> anyhow::Result<()> {
    let (sender, receiver) = unbounded();

    // spawn the coroutines on the executor
    let font_loader = ex.spawn(load_font());
    let timer = match format {
        cpal::SampleFormat::F32 => ex.spawn(timer::<f32>(sender, device, config)),
        cpal::SampleFormat::U16 => ex.spawn(timer::<u16>(sender, device, config)),
        cpal::SampleFormat::I16 => ex.spawn(timer::<i16>(sender, device, config)),
    };
    let x11 = ex.spawn(x_process(receiver));

    // run all three future simultaneously
    font_loader.await?;
    let (tres, xres) = future::zip(timer, x11).await;
    xres.and(tres)
}

fn main() -> anyhow::Result<()> {
    // spawn the executor
    let (signal, shutdown) = unbounded::<()>();
    let ex = Executor::new();

    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();
    let config = device.default_output_config().unwrap();

    let res = easy_parallel::Parallel::new()
        .each(0..2, |_| smol::block_on(ex.run(shutdown.recv())))
        .finish(|| {
            let res = smol::block_on(entry(&ex, device, config.sample_format(), config.into()));
            mem::drop(signal);
            res
        })
        .1;
    res
}
