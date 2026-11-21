use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct LoadedWavTrack {
    pub stream_idx: u32,
    pub path: PathBuf,
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub total_frames: usize,
    pub duration_secs: f32,
    pub samples: Vec<i16>,
    pub waveform_peaks: Vec<f32>,
    pub waveform_rms: Vec<f32>,
    pub overall_peak: f32,
}

impl LoadedWavTrack {
    pub fn from_wav_file(stream_idx: u32, path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("Read error: {e}"))?;
        if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err("Not a valid RIFF/WAVE file".into());
        }
        let mut off = 12usize;
        let mut channels = 1u16;
        let mut sample_rate = 48_000u32;
        let mut bits_per_sample = 16u16;
        let mut data_slice: Option<&[u8]> = None;

        while off + 8 <= bytes.len() {
            let id = &bytes[off..off + 4];
            let sz = u32::from_le_bytes([bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]]) as usize;
            let body_start = off + 8;
            let body_end = (body_start + sz).min(bytes.len());
            if id == b"fmt " && body_end - body_start >= 16 {
                let b = &bytes[body_start..body_end];
                channels = u16::from_le_bytes([b[2], b[3]]).max(1);
                sample_rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]).max(1);
                bits_per_sample = u16::from_le_bytes([b[14], b[15]]);
            } else if id == b"data" {
                data_slice = Some(&bytes[body_start..body_end]);
                break;
            }
            off = body_start + sz + (sz & 1);
        }

        if bits_per_sample != 16 {
            return Err(format!("Unsupported WAV bit depth: {bits_per_sample}"));
        }
        let raw = data_slice.ok_or_else(|| "Missing WAV data chunk".to_string())?;
        let mut samples = Vec::with_capacity(raw.len() / 2);
        for pair in raw.chunks_exact(2) {
            samples.push(i16::from_le_bytes([pair[0], pair[1]]));
        }

        let ch = channels as usize;
        let total_frames = samples.len() / ch;
        let duration_secs = total_frames as f32 / sample_rate as f32;

        const BINS: usize = 200;
        let mut waveform_peaks = vec![0.0f32; BINS];
        let mut waveform_rms = vec![0.0f32; BINS];
        let mut overall_peak = 0.0f32;

        if total_frames > 0 {
            for b in 0..BINS {
                let f0 = b * total_frames / BINS;
                let f1 = ((b + 1) * total_frames / BINS).max(f0 + 1).min(total_frames);
                let mut peak = 0.0f32;
                let mut sum_sq = 0.0f64;
                let mut cnt = 0usize;
                for f in f0..f1 {
                    let base = f * ch;
                    let mut frame_abs = 0.0f32;
                    for c in 0..ch {
                        let s = (samples[base + c] as f32 / 32768.0).abs();
                        if s > frame_abs {
                            frame_abs = s;
                        }
                        sum_sq += (s as f64) * (s as f64);
                        cnt += 1;
                    }
                    if frame_abs > peak {
                        peak = frame_abs;
                    }
                }
                waveform_peaks[b] = peak;
                waveform_rms[b] = if cnt > 0 { (sum_sq / cnt as f64).sqrt() as f32 } else { 0.0 };
                if peak > overall_peak {
                    overall_peak = peak;
                }
            }
        }

        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        Ok(Self {
            stream_idx,
            path: path.to_path_buf(),
            name,
            sample_rate,
            channels,
            total_frames,
            duration_secs,
            samples,
            waveform_peaks,
            waveform_rms,
            overall_peak,
        })
    }

    /// Computes instantaneous peak/RMS loudness around `frame` (within ~40ms window).
    pub fn loudness_at_frame(&self, frame: usize) -> (f32, f32) {
        if self.total_frames == 0 {
            return (0.0, 0.0);
        }
        let ch = self.channels as usize;
        let win = (self.sample_rate as usize / 25).max(64);
        let f0 = frame.saturating_sub(win / 2);
        let f1 = (frame + win / 2).min(self.total_frames);
        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut cnt = 0usize;
        for f in f0..f1 {
            let base = f * ch;
            for c in 0..ch {
                let s = (self.samples[base + c] as f32 / 32768.0).abs();
                if s > peak {
                    peak = s;
                }
                sum_sq += (s as f64) * (s as f64);
                cnt += 1;
            }
        }
        let rms = if cnt > 0 { (sum_sq / cnt as f64).sqrt() as f32 } else { 0.0 };
        (peak, rms)
    }
}

#[cfg(windows)]
mod backend {
    use std::mem::size_of;

    const WAVE_MAPPER: u32 = 0xFFFF_FFFF;
    const TIME_BYTES: u32 = 0x0004;
    const WHDR_DONE: u32 = 0x0000_0001;

    #[repr(C)]
    struct WAVEFORMATEX {
        w_format_tag: u16,
        n_channels: u16,
        n_samples_per_sec: u32,
        n_avg_bytes_per_sec: u32,
        n_block_align: u16,
        w_bits_per_sample: u16,
        cb_size: u16,
    }

    #[repr(C)]
    struct WAVEHDR {
        lp_data: *mut u8,
        dw_buffer_length: u32,
        dw_bytes_recorded: u32,
        dw_user: usize,
        dw_flags: u32,
        dw_loops: u32,
        lp_next: usize,
        reserved: usize,
    }

    #[repr(C)]
    struct MMTIME {
        w_type: u32,
        u: [u32; 2],
    }

    #[link(name = "winmm")]
    extern "system" {
        fn waveOutOpen(phwo: *mut isize, u_device_id: u32, pwfx: *const WAVEFORMATEX, dw_callback: usize, dw_instance: usize, fdw_open: u32) -> u32;
        fn waveOutClose(hwo: isize) -> u32;
        fn waveOutPrepareHeader(hwo: isize, pwh: *mut WAVEHDR, cbwh: u32) -> u32;
        fn waveOutUnprepareHeader(hwo: isize, pwh: *mut WAVEHDR, cbwh: u32) -> u32;
        fn waveOutWrite(hwo: isize, pwh: *mut WAVEHDR, cbwh: u32) -> u32;
        fn waveOutPause(hwo: isize) -> u32;
        fn waveOutRestart(hwo: isize) -> u32;
        fn waveOutReset(hwo: isize) -> u32;
        fn waveOutGetPosition(hwo: isize, pmmt: *mut MMTIME, cbmmt: u32) -> u32;
        fn waveOutSetVolume(hwo: isize, dw_volume: u32) -> u32;
    }

    pub struct WaveOutHandle {
        hwo: isize,
        hdr: Box<WAVEHDR>,
        _pcm_buf: Vec<u8>,
        block_align: usize,
        pub start_frame: usize,
        pub total_frames: usize,
    }

    // Safety: WaveOutHandle owns its pinned heap WAVEHDR and PCM Vec<u8> and is only accessed via &mut on the UI thread or Drop.
    unsafe impl Send for WaveOutHandle {}

    impl WaveOutHandle {
        pub fn play_from(samples: &[i16], sample_rate: u32, channels: u16, start_frame: usize, volume: f32) -> Result<Self, String> {
            let ch = channels.max(1) as usize;
            let block_align = ch * 2;
            let total_frames = samples.len() / ch;
            let start = start_frame.min(total_frames);
            let slice = &samples[start * ch..];
            let mut pcm_buf = Vec::with_capacity(slice.len() * 2);
            for &s in slice {
                pcm_buf.extend_from_slice(&s.to_le_bytes());
            }
            if pcm_buf.is_empty() {
                pcm_buf.extend_from_slice(&[0u8; 4]);
            }

            let wfx = WAVEFORMATEX {
                w_format_tag: 1,
                n_channels: channels.max(1),
                n_samples_per_sec: sample_rate.max(1),
                n_avg_bytes_per_sec: sample_rate.max(1) * block_align as u32,
                n_block_align: block_align as u16,
                w_bits_per_sample: 16,
                cb_size: 0,
            };

            let mut hwo: isize = 0;
            let rc = unsafe { waveOutOpen(&mut hwo, WAVE_MAPPER, &wfx, 0, 0, 0) };
            if rc != 0 || hwo == 0 {
                return Err(format!("waveOutOpen failed (code {rc})"));
            }

            let mut hdr = Box::new(WAVEHDR {
                lp_data: pcm_buf.as_mut_ptr(),
                dw_buffer_length: pcm_buf.len() as u32,
                dw_bytes_recorded: 0,
                dw_user: 0,
                dw_flags: 0,
                dw_loops: 0,
                lp_next: 0,
                reserved: 0,
            });

            let rc_prep = unsafe { waveOutPrepareHeader(hwo, &mut *hdr, size_of::<WAVEHDR>() as u32) };
            if rc_prep != 0 {
                unsafe { waveOutClose(hwo); }
                return Err(format!("waveOutPrepareHeader failed (code {rc_prep})"));
            }

            let mut handle = Self {
                hwo,
                hdr,
                _pcm_buf: pcm_buf,
                block_align,
                start_frame: start,
                total_frames,
            };
            handle.set_volume(volume);

            let rc_write = unsafe { waveOutWrite(handle.hwo, &mut *handle.hdr, size_of::<WAVEHDR>() as u32) };
            if rc_write != 0 {
                return Err(format!("waveOutWrite failed (code {rc_write})"));
            }
            Ok(handle)
        }

        pub fn set_volume(&mut self, volume: f32) {
            let v = (volume.clamp(0.0, 1.0) * 65535.0) as u32;
            let dword = (v << 16) | v;
            unsafe { waveOutSetVolume(self.hwo, dword); }
        }

        pub fn pause(&mut self) {
            unsafe { waveOutPause(self.hwo); }
        }

        pub fn resume(&mut self) {
            unsafe { waveOutRestart(self.hwo); }
        }

        pub fn current_frame(&mut self) -> (usize, bool) {
            let done = (self.hdr.dw_flags & WHDR_DONE) != 0;
            let mut mmt = MMTIME { w_type: TIME_BYTES, u: [0, 0] };
            let rc = unsafe { waveOutGetPosition(self.hwo, &mut mmt, size_of::<MMTIME>() as u32) };
            let played_frames = if rc == 0 {
                (mmt.u[0] as usize) / self.block_align
            } else {
                0
            };
            let cur = (self.start_frame + played_frames).min(self.total_frames);
            (cur, done || cur >= self.total_frames)
        }
    }

    impl Drop for WaveOutHandle {
        fn drop(&mut self) {
            unsafe {
                waveOutReset(self.hwo);
                waveOutUnprepareHeader(self.hwo, &mut *self.hdr, size_of::<WAVEHDR>() as u32);
                waveOutClose(self.hwo);
            }
        }
    }
}

#[cfg(not(windows))]
mod backend {
    use std::time::Instant;

    pub struct WaveOutHandle {
        start_frame: usize,
        total_frames: usize,
        sample_rate: u32,
        started_at: Instant,
        paused_frame: Option<usize>,
    }

    impl WaveOutHandle {
        pub fn play_from(samples: &[i16], sample_rate: u32, channels: u16, start_frame: usize, _volume: f32) -> Result<Self, String> {
            let ch = channels.max(1) as usize;
            let total_frames = samples.len() / ch;
            Ok(Self {
                start_frame: start_frame.min(total_frames),
                total_frames,
                sample_rate: sample_rate.max(1),
                started_at: Instant::now(),
                paused_frame: None,
            })
        }

        pub fn set_volume(&mut self, _volume: f32) {}
        pub fn pause(&mut self) {
            let (cur, _) = self.current_frame();
            self.paused_frame = Some(cur);
        }
        pub fn resume(&mut self) {
            if let Some(f) = self.paused_frame.take() {
                self.start_frame = f;
                self.started_at = Instant::now();
            }
        }
        pub fn current_frame(&mut self) -> (usize, bool) {
            if let Some(f) = self.paused_frame {
                return (f, f >= self.total_frames);
            }
            let elapsed = self.started_at.elapsed().as_secs_f64();
            let cur = (self.start_frame + (elapsed * self.sample_rate as f64) as usize).min(self.total_frames);
            (cur, cur >= self.total_frames)
        }
    }
}

pub struct AudioPlayerState {
    pub tracks: Vec<LoadedWavTrack>,
    pub active_idx: Option<usize>,
    pub playing: bool,
    pub paused: bool,
    pub cursor_frame: usize,
    pub volume: f32,
    pub error: Option<String>,
    handle: Option<backend::WaveOutHandle>,
}

impl Default for AudioPlayerState {
    fn default() -> Self {
        Self {
            tracks: Vec::new(),
            active_idx: None,
            playing: false,
            paused: false,
            cursor_frame: 0,
            volume: 0.85,
            error: None,
            handle: None,
        }
    }
}

impl AudioPlayerState {
    pub fn load_streams(&mut self, streams: &[(u32, PathBuf)]) {
        self.stop();
        self.tracks.clear();
        self.active_idx = None;
        self.error = None;
        for &(idx, ref p) in streams {
            if let Ok(t) = LoadedWavTrack::from_wav_file(idx, p) {
                self.tracks.push(t);
            }
        }
    }

    pub fn play_track(&mut self, idx: usize, start_frame: usize) {
        let Some(track) = self.tracks.get(idx) else { return };
        self.handle = None;
        self.error = None;
        let clamped = start_frame.min(track.total_frames);
        match backend::WaveOutHandle::play_from(
            &track.samples,
            track.sample_rate,
            track.channels,
            clamped,
            self.volume,
        ) {
            Ok(h) => {
                self.handle = Some(h);
                self.active_idx = Some(idx);
                self.cursor_frame = clamped;
                self.playing = true;
                self.paused = false;
            }
            Err(e) => {
                self.error = Some(e);
                self.playing = false;
                self.paused = false;
            }
        }
    }

    pub fn toggle_play_pause(&mut self) {
        if self.playing && !self.paused {
            if let Some(h) = self.handle.as_mut() {
                let (cur, _) = h.current_frame();
                self.cursor_frame = cur;
                h.pause();
                self.paused = true;
            }
        } else if self.paused {
            if let Some(h) = self.handle.as_mut() {
                h.resume();
                self.paused = false;
            }
        } else if let Some(idx) = self.active_idx.or_else(|| (!self.tracks.is_empty()).then_some(0)) {
            let start = if let Some(t) = self.tracks.get(idx) {
                if self.cursor_frame >= t.total_frames { 0 } else { self.cursor_frame }
            } else {
                0
            };
            self.play_track(idx, start);
        }
    }

    pub fn seek_to_frame(&mut self, frame: usize) {
        let Some(idx) = self.active_idx else { return };
        let Some(track) = self.tracks.get(idx) else { return };
        let target = frame.min(track.total_frames);
        self.cursor_frame = target;
        if self.playing && !self.paused {
            self.play_track(idx, target);
        } else {
            self.handle = None;
            self.playing = false;
            self.paused = false;
        }
    }

    pub fn seek_by_secs(&mut self, delta_secs: f32) {
        let Some(idx) = self.active_idx else { return };
        let Some((sample_rate, total_frames)) = self.tracks.get(idx).map(|t| (t.sample_rate, t.total_frames)) else { return };
        let cur = self.current_frame();
        let delta_frames = (delta_secs * sample_rate as f32).round() as isize;
        let next = (cur as isize + delta_frames).clamp(0, total_frames as isize) as usize;
        self.seek_to_frame(next);
    }

    pub fn stop(&mut self) {
        self.handle = None;
        self.playing = false;
        self.paused = false;
        self.cursor_frame = 0;
    }

    pub fn current_frame(&mut self) -> usize {
        if self.playing && !self.paused {
            if let Some(h) = self.handle.as_mut() {
                let (cur, done) = h.current_frame();
                self.cursor_frame = cur;
                if done {
                    self.handle = None;
                    self.playing = false;
                    self.paused = false;
                }
            }
        }
        self.cursor_frame
    }

    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        if let Some(h) = self.handle.as_mut() {
            h.set_volume(self.volume);
        }
    }
}

fn fmt_time(secs: f32) -> String {
    let s = secs.max(0.0);
    let mins = (s / 60.0).floor() as u32;
    let rem = s - (mins as f32) * 60.0;
    format!("{mins:02}:{rem:04.1}")
}

pub fn render_awc_audio_view(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    player: &mut AudioPlayerState,
    status_headline: &str,
) {
    let cur_frame = player.current_frame();
    if player.playing && !player.paused {
        ctx.request_repaint();
    }

    // Header bar
    ui.horizontal(|ui| {
        ui.label(RichText::new(status_headline).strong().color(Color32::WHITE));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(format!("{} stream(s)", player.tracks.len()))
                    .size(11.5)
                    .color(Color32::from_rgb(175, 160, 200)),
            );
        });
    });
    ui.add_space(4.0);
    ui.separator();

    // Reserve fixed height at the bottom for the audio player overlay bar
    let player_height = 128.0;
    let list_height = (ui.available_height() - player_height - 8.0).max(80.0);

    // Full-width stream list
    let mut clicked_track: Option<usize> = None;
    egui::ScrollArea::vertical()
        .id_salt("awc_stream_list")
        .max_height(list_height)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            for (i, t) in player.tracks.iter().enumerate() {
                let is_active = player.active_idx == Some(i);
                let is_playing = is_active && player.playing && !player.paused;
                let row_bg = if is_active {
                    Color32::from_rgb(56, 30, 82)
                } else if i % 2 == 0 {
                    Color32::from_rgb(24, 15, 36)
                } else {
                    Color32::from_rgb(29, 18, 43)
                };
                let stroke = if is_active {
                    Stroke::new(1.0_f32, Color32::from_rgb(255, 168, 28))
                } else {
                    Stroke::NONE
                };

                let frame_resp = egui::Frame::new()
                    .fill(row_bg)
                    .stroke(stroke)
                    .corner_radius(CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(8, 5))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.horizontal(|ui| {
                            let btn_label = if is_playing { "Pause" } else { "Play" };
                            let btn = egui::Button::new(
                                RichText::new(btn_label)
                                    .strong()
                                    .color(if is_active { Color32::BLACK } else { Color32::WHITE }),
                            )
                            .fill(if is_active {
                                Color32::from_rgb(255, 168, 28)
                            } else {
                                Color32::from_rgb(52, 34, 74)
                            })
                            .min_size(Vec2::new(48.0, 22.0));

                            if ui.add(btn).clicked() {
                                clicked_track = Some(i);
                            }

                            ui.add_space(6.0);
                            ui.label(
                                RichText::new(format!("Stream #{}", t.stream_idx))
                                    .strong()
                                    .color(if is_active {
                                        Color32::from_rgb(255, 168, 28)
                                    } else {
                                        Color32::from_rgb(225, 215, 240)
                                    }),
                            );
                            ui.label(
                                RichText::new(&t.name)
                                    .monospace()
                                    .size(11.5)
                                    .color(Color32::from_rgb(165, 150, 190)),
                            );

                            // Right-aligned metadata + inline peak meter bar
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(64.0, 10.0), Sense::hover());
                                let p = ui.painter();
                                p.rect_filled(bar_rect, CornerRadius::same(2), Color32::from_rgb(40, 26, 58));
                                let fill_w = (bar_rect.width() * t.overall_peak.clamp(0.0, 1.0)).max(2.0);
                                let fill_rect = Rect::from_min_size(bar_rect.min, Vec2::new(fill_w, bar_rect.height()));
                                let peak_col = if t.overall_peak > 0.85 {
                                    Color32::from_rgb(255, 72, 165)
                                } else {
                                    Color32::from_rgb(255, 168, 28)
                                };
                                p.rect_filled(fill_rect, CornerRadius::same(2), peak_col);

                                ui.add_space(8.0);
                                let ch_str = if t.channels == 2 { "Stereo" } else { "Mono" };
                                ui.label(
                                    RichText::new(format!("{} · {} kHz · {}", fmt_time(t.duration_secs), t.sample_rate / 1000, ch_str))
                                        .size(11.5)
                                        .color(Color32::from_rgb(175, 200, 225)),
                                );
                            });
                        });
                    });

                if frame_resp.response.interact(Sense::click()).clicked() {
                    clicked_track = Some(i);
                }
                ui.add_space(2.0);
            }
        });

    if let Some(i) = clicked_track {
        if player.active_idx == Some(i) && (player.playing || player.paused) {
            player.toggle_play_pause();
        } else {
            player.play_track(i, 0);
        }
    }

    ui.add_space(6.0);

    // Bottom Audio Player Overlay Bar with Loudness Playbar + Seek Controls
    egui::Frame::new()
        .fill(Color32::from_rgb(26, 15, 40))
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(105, 62, 145)))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(10.0)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            let active_track = player.active_idx.and_then(|i| player.tracks.get(i)).cloned();

            // Row 1: Track info + transport / seek controls + volume
            ui.horizontal(|ui| {
                if let Some(ref t) = active_track {
                    ui.label(
                        RichText::new(format!("Stream #{} · {}", t.stream_idx, t.name))
                            .strong()
                            .color(Color32::from_rgb(255, 168, 28)),
                    );
                } else {
                    ui.label(RichText::new("Select a stream above to play").color(Color32::from_rgb(165, 150, 190)));
                }

                ui.add_space(12.0);
                let can_prev = player.active_idx.map(|i| i > 0).unwrap_or(false);
                if ui.add_enabled(can_prev, egui::Button::new("|<")).on_hover_text("Previous stream").clicked() {
                    if let Some(i) = player.active_idx {
                        player.play_track(i - 1, 0);
                    }
                }
                if ui.add_enabled(active_track.is_some(), egui::Button::new("-5s")).on_hover_text("Seek back 5s").clicked() {
                    player.seek_by_secs(-5.0);
                }
                let play_lbl = if player.playing && !player.paused { "Pause" } else { "Play" };
                let play_btn = egui::Button::new(RichText::new(play_lbl).strong().color(Color32::BLACK))
                    .fill(Color32::from_rgb(255, 168, 28));
                if ui.add_enabled(!player.tracks.is_empty(), play_btn).clicked() {
                    player.toggle_play_pause();
                }
                if ui.add_enabled(player.playing || player.paused, egui::Button::new("Stop")).clicked() {
                    player.stop();
                }
                if ui.add_enabled(active_track.is_some(), egui::Button::new("+5s")).on_hover_text("Seek forward 5s").clicked() {
                    player.seek_by_secs(5.0);
                }
                let can_next = player
                    .active_idx
                    .map(|i| i + 1 < player.tracks.len())
                    .unwrap_or(false);
                if ui.add_enabled(can_next, egui::Button::new(">|")).on_hover_text("Next stream").clicked() {
                    if let Some(i) = player.active_idx {
                        player.play_track(i + 1, 0);
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut vol = player.volume;
                    if ui.add(egui::Slider::new(&mut vol, 0.0..=1.0).show_value(false).custom_formatter(|v, _| format!("{:.0}%", v * 100.0))).changed() {
                        player.set_volume(vol);
                    }
                    ui.label(RichText::new(format!("Vol {:.0}%", player.volume * 100.0)).size(11.5).color(Color32::from_rgb(200, 190, 220)));
                });
            });

            ui.add_space(6.0);

            // Row 2: Loudness / Waveform Playbar (interactive seek bar!)
            let bar_w = ui.available_width();
            let (wave_rect, wave_resp) = ui.allocate_exact_size(Vec2::new(bar_w, 48.0), Sense::click_and_drag());
            let painter = ui.painter_at(wave_rect);
            painter.rect_filled(wave_rect, CornerRadius::same(4), Color32::from_rgb(16, 10, 26));
            painter.rect_stroke(wave_rect, CornerRadius::same(4), Stroke::new(1.0_f32, Color32::from_rgb(65, 42, 92)), egui::StrokeKind::Inside);

            if let Some(ref t) = active_track {
                let frac = if t.total_frames > 0 {
                    (cur_frame as f32 / t.total_frames as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let bins = t.waveform_peaks.len().max(1);
                let step = wave_rect.width() / bins as f32;
                let bar_draw_w = (step * 0.72).max(1.5);
                let mid_y = wave_rect.center().y;
                let max_half_h = (wave_rect.height() - 8.0) * 0.5;

                for (b, (&peak, &rms)) in t.waveform_peaks.iter().zip(t.waveform_rms.iter()).enumerate() {
                    let bx = wave_rect.min.x + b as f32 * step + step * 0.14;
                    let bin_frac = (b as f32 + 0.5) / bins as f32;
                    let played = bin_frac <= frac;

                    let h_peak = (peak * max_half_h).max(1.5);
                    let h_rms = (rms * max_half_h).max(1.0);

                    let peak_col = if played {
                        if peak > 0.85 {
                            Color32::from_rgb(255, 72, 165)
                        } else {
                            Color32::from_rgb(255, 168, 28)
                        }
                    } else {
                        Color32::from_rgb(76, 54, 105)
                    };
                    let rms_col = if played {
                        Color32::from_rgb(255, 215, 120)
                    } else {
                        Color32::from_rgb(104, 78, 138)
                    };

                    painter.rect_filled(
                        Rect::from_min_max(Pos2::new(bx, mid_y - h_peak), Pos2::new(bx + bar_draw_w, mid_y + h_peak)),
                        CornerRadius::same(1),
                        peak_col,
                    );
                    painter.rect_filled(
                        Rect::from_min_max(Pos2::new(bx, mid_y - h_rms), Pos2::new(bx + bar_draw_w, mid_y + h_rms)),
                        CornerRadius::same(1),
                        rms_col,
                    );
                }

                // Playhead line + handle
                let head_x = wave_rect.min.x + wave_rect.width() * frac;
                painter.line_segment(
                    [Pos2::new(head_x, wave_rect.min.y + 1.0), Pos2::new(head_x, wave_rect.max.y - 1.0)],
                    Stroke::new(2.0_f32, Color32::WHITE),
                );
                painter.circle_filled(Pos2::new(head_x, wave_rect.min.y + 5.0), 4.0, Color32::from_rgb(255, 168, 28));

                // Click or drag to seek
                if (wave_resp.clicked() || wave_resp.dragged()) && t.total_frames > 0 {
                    if let Some(pos) = wave_resp.interact_pointer_pos() {
                        let seek_frac = ((pos.x - wave_rect.min.x) / wave_rect.width()).clamp(0.0, 1.0);
                        let target_frame = (seek_frac * t.total_frames as f32) as usize;
                        player.seek_to_frame(target_frame);
                    }
                }
                if let Some(hover_pos) = wave_resp.hover_pos() {
                    let hfrac = ((hover_pos.x - wave_rect.min.x) / wave_rect.width()).clamp(0.0, 1.0);
                    let hsecs = hfrac * t.duration_secs;
                    wave_resp.on_hover_text(format!("Seek to {}", fmt_time(hsecs)));
                }
            } else {
                painter.text(
                    wave_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Waveform / Loudness Playbar",
                    egui::FontId::proportional(12.0),
                    Color32::from_rgb(120, 100, 150),
                );
            }

            ui.add_space(4.0);

            // Row 3: Time readout + Live Loudness Level Meter
            ui.horizontal(|ui| {
                if let Some(ref t) = active_track {
                    let cur_secs = cur_frame as f32 / t.sample_rate as f32;
                    ui.label(
                        RichText::new(format!("{} / {}", fmt_time(cur_secs), fmt_time(t.duration_secs)))
                            .monospace()
                            .size(11.5)
                            .color(Color32::from_rgb(225, 215, 240)),
                    );

                    let (live_peak, live_rms) = if player.playing && !player.paused {
                        t.loudness_at_frame(cur_frame)
                    } else {
                        (0.0, 0.0)
                    };
                    let db = if live_peak > 1e-5 { 20.0 * live_peak.log10() } else { -96.0 };

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (meter_rect, _) = ui.allocate_exact_size(Vec2::new(140.0, 10.0), Sense::hover());
                        let mp = ui.painter();
                        mp.rect_filled(meter_rect, CornerRadius::same(2), Color32::from_rgb(18, 12, 28));
                        let mw = (meter_rect.width() * live_peak.clamp(0.0, 1.0)).max(0.0);
                        if mw > 0.5 {
                            let col = if live_peak > 0.85 {
                                Color32::from_rgb(255, 72, 165)
                            } else if live_peak > 0.5 {
                                Color32::from_rgb(255, 168, 28)
                            } else {
                                Color32::from_rgb(100, 220, 140)
                            };
                            mp.rect_filled(
                                Rect::from_min_size(meter_rect.min, Vec2::new(mw, meter_rect.height())),
                                CornerRadius::same(2),
                                col,
                            );
                        }
                        let rw = (meter_rect.width() * live_rms.clamp(0.0, 1.0)).max(0.0);
                        if rw > 0.5 {
                            mp.rect_filled(
                                Rect::from_min_size(meter_rect.min, Vec2::new(rw, meter_rect.height())),
                                CornerRadius::same(2),
                                Color32::from_rgba_unmultiplied(255, 255, 255, 140),
                            );
                        }
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(format!("Level: {:5.1} dB ({:.0}%)", db, live_peak * 100.0))
                                .monospace()
                                .size(11.0)
                                .color(Color32::from_rgb(180, 170, 205)),
                        );
                    });
                } else if let Some(ref err) = player.error {
                    ui.label(RichText::new(err).color(Color32::from_rgb(255, 120, 120)));
                }
            });
        });
}
