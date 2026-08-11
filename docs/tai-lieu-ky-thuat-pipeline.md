# Tài liệu kỹ thuật — Pipeline phục hồi âm thanh V2 → V3

**Ngày:** 11/08/2026
**Phạm vi:** mô tả toàn bộ đường đi của tín hiệu từ file upload đến file xuất, model dùng ở từng bước, kỹ thuật xử lý, và trạng thái hiện tại của từng thành phần.
**Nguồn:** viết trực tiếp từ mã nguồn trong repo (`midi_repair/`, `suno_restore/`, `full_pipeline.py`), không phải từ tài liệu thiết kế cũ.

> **Lưu ý về tài liệu cũ:** `docs/design/2026-08-10-merged-pipeline.md` mô tả V3 có "damage gates", "wet/dry blend" và "verification stage". **Những cơ chế đó đã bị gỡ khỏi mã nguồn.** Hiện tại V3 chạy mỗi bước *vô điều kiện và ở cường độ đầy đủ* nếu bước đó được bật (xem `suno_restore/config.py` và docstring của `full_pipeline.py`). Tài liệu này mô tả đúng trạng thái mã nguồn hiện tại.

---

## 1. Tổng quan kiến trúc

Hệ thống gồm **hai pipeline nối tiếp**, mỗi cái giải quyết một loại hư hỏng khác nhau:

```
File upload (MP3/WAV từ Suno)
        │
        ▼
┌───────────────────────────────────────────────┐
│ VERSION 2 — midi_repair                       │
│ Sửa MỘT vùng hỏng cục bộ bằng MIDI            │
│ (tách stem → chép nốt → phát hiện → tái tạo)  │
└───────────────────────────────────────────────┘
        │  v2/<tên>_v2.wav
        ▼
┌───────────────────────────────────────────────┐
│ VERSION 3 — suno_restore                      │
│ Sửa 3 lỗi TOÀN BỘ file                        │
│  Bước 1: Tempo (trục thời gian)               │
│  Bước 2: Denoise (khử ồn)                     │
│  Bước 3: Bandwidth (mở rộng dải tần)          │
└───────────────────────────────────────────────┘
        │  v3/<tên>_restored.wav
        ▼
   quality.py — đo đạc, so sánh
```

**Tại sao thứ tự là V2 → V3 (không đảo ngược):** bộ phát hiện của V2 được hiệu chuẩn (calibrate) trên phổ tần của file gốc. Nếu chạy V3 trước, V3 sẽ thay đổi chính cái phổ mà bộ phát hiện của V2 cần đo → ngưỡng phát hiện mất hiệu lực.

Điểm vào chính: `full_pipeline.restore_from_stem()`. Giao diện: `merged_app.py` (Streamlit).

---

## 2. Nền tảng chung — I/O và tái lấy mẫu (`suno_restore/audio_io.py`)

Đây là lớp nền mọi bước đều dùng. Có ba quyết định kỹ thuật quan trọng:

### 2.1 Giải mã bằng ffmpeg, không dùng soundfile/librosa

File stem Suno xuất ra là **MP3 bitrate rất thấp (32–36 kbps), header VBR không đáng tin**. `libsndfile 1.2.2` **cắt cụt file trong im lặng**:

| File | Độ dài thật | libsndfile đọc được |
|---|---|---|
| Bass stem | 294 s | **87 s** |
| Backing vocal stem | 294 s | **68 s** |

Nguy hiểm nhất: `sf.info` báo cáo *cùng cái độ dài sai đó*, nên không có bước nào phía sau phát hiện được. Vì vậy toàn bộ giải mã đi qua `ffmpeg` (subprocess, xuất `f32le` raw), `ffprobe` để đọc sample rate và số kênh.

### 2.2 Sample rate "đi theo nguồn"

`RESTORE_OUTPUT_SAMPLE_RATE=0` (mặc định) nghĩa là **giữ nguyên tần số lấy mẫu của file gốc**. Lý do: mỗi lần resample là một bộ lọc. Ép mọi thứ về 48 kHz khiến file 44.1 kHz bị resample hai lần (vào 44.1 → ra 48) mà không ai yêu cầu.

Resample dùng **`soxr`** (chất lượng cao) chứ không phải bộ resample của librosa.

### 2.3 Hợp đồng về số kênh (channel contract)

`enforce_channel_count = True`. Sau **mỗi** bước, `pipeline.check_integrity()` so số kênh vào/ra; lệch là ném `IntegrityError` ngay.

Đây không phải lựa chọn chính sách mà là **chốt chặn cho một lỗi có thật**: model denoise có thể ghi ra file mono, và phiên bản trước lấy luôn số kênh của file đó → **biến mọi stem stereo thành mono mà vẫn báo cáo thành công**. Sập kênh là không thể phục hồi và không hiện lên trong bất kỳ chỉ số nào khác.

**Xuất file:** WAV **PCM 24-bit** (`save_audio`).

---

## 3. VERSION 2 — `midi_repair` (sửa cục bộ bằng MIDI)

### 3.1 V2 làm gì

V2 giả định: file đầu vào **có thể** chứa **một vùng liên tục bị hỏng** (nghe như nhiễu, phát hiện được bằng spectral flatness). Nó không sửa toàn bài — nó chỉ tái tạo đúng một đoạn.

Thông số nội bộ: `SR = 22050 Hz`, **mono**, `HOP_LENGTH = 512`.

### 3.2 Các bước và model

| Bước | Việc làm | Model / thư viện |
|---|---|---|
| 1a | Tách stem | **BS-RoFormer** (`bs-roformer-infer`, `DEFAULT_MODEL`) |
| 1b | Chuyển audio → MIDI | **MuScriptor** `TranscriptionModel` (`large` ~1.37B tham số; tự hạ xuống `medium` nếu GPU < 12 GB) |
| 1c–1e | Phân tích key + căn chỉnh | **music21** (phân tích giọng), **librosa** `chroma_cqt`, **dtw-python** (DTW alignment) |
| 2a | Phát hiện vùng hỏng | **librosa** `spectral_flatness` + quy tắc ngưỡng |
| 2b | Tái tạo MIDI cho vùng hỏng | **stanford-crfm/music-small-800k** (AMT) qua thư viện **anticipation** |
| 3a | Render lại từ nốt "donor" | **librosa** `pitch_shift` + `time_stretch` |
| 3b | Ghép lại + kiểm tra lại | Crossfade equal-power (cos/sin), 50 ms |

**Chi tiết bộ phát hiện (`detect_flagged_region`):**
- Hiệu chuẩn từ nguyên mẫu rewrite-v2: audio sạch chưa bao giờ vượt flatness ~0.20; vùng bị chèn nhiễu nằm ở 0.26–0.42. Do đó `FLATNESS_LOW = 0.15`, `FLATNESS_HIGH = 0.35`.
- Ngưỡng: `max(mean + 1.5×std, 0.4)`; gộp các đoạn cách nhau < 1.0 s; tối thiểu 1.0 s; đệm 0.25 s mỗi bên.
- **Chỉ lấy đúng một vùng** — vùng dài nhất.

**Chi tiết tái tạo (`render_window`):** với mỗi nốt MIDI cần tái tạo, tìm một nốt "donor" *có thật trong chính file đó* (ưu tiên trùng cao độ, thời lượng gần 0.3 s), dịch cao độ nếu cần, kéo giãn về đúng trường độ, rồi đặt vào vị trí. Tức là **âm sắc được giữ nguyên vì vật liệu lấy từ chính bản thu**, không phải sinh ra từ synth.

### 3.3 Hai cơ chế an toàn quan trọng của V2

**(a) Rollback theo rủi ro đo được.** Sau khi ghép, V2 đo lại chỉ số rủi ro trong đúng vùng đó. Nếu `risk_after >= risk_before` → **vứt bản sửa, trả lại file gốc**. V2 không bao giờ giao một bản sửa mà nó không chứng minh được là tốt hơn.

**(b) Chỉ áp dụng *delta*, không thay thế file** (`preserve_source_layout`). V2 phân tích ở **22.05 kHz mono**, nhưng file gốc có thể là 44.1 kHz stereo. Nếu đưa thẳng bản render của V2 sang V3 thì **mất stereo và mất toàn bộ nội dung trên 11.025 kHz**. Thay vào đó:

```
delta = (audio_đã_sửa − audio_phân_tích)      # đều ở 22.05 kHz mono
delta_gốc = resample(delta, 22050 → sr_gốc)
kết_quả = audio_gốc + delta_gốc               # cộng vào MỌI kênh
```

Cộng cùng một delta vào cả hai kênh **giữ nguyên hiệu số stereo (L−R)**. Mọi mẫu ngoài vùng sửa vẫn đến từ bản giải mã gốc.

### 3.4 Vì sao trọng tâm chuyển sang V3

V2 hoạt động đúng như thiết kế, nhưng **chi phí/lợi ích không thuận lợi trong khung thời gian có hạn**:

1. **Chi phí tính toán rất lớn:** V2 phải nạp **ba model nặng** (BS-RoFormer + MuScriptor ~1.37B + AMT). Trên card 8 GB, V2 và V3 **không thể cùng nằm trên GPU** — phải gọi `midi_repair.unload_models()` trong khối `finally` tại ranh giới V2→V3.
2. **Phạm vi rất hẹp:** cả pipeline đó chỉ để sửa **đúng một vùng**. Nếu không có vùng nào vượt ngưỡng, V2 **trả file về nguyên vẹn** — toàn bộ thời gian chạy 3 model là công cốc.
3. **Vấn đề thực tế của material lại nằm chỗ khác.** Đo trên bộ stem tham chiếu (xem `docs/restoration-requirements.md`):

| Lỗi | Kết luận đo được | Ưu tiên |
|---|---|---|
| **Mất dải cao** — không có nội dung trên ~16–18.5 kHz | **Có thật và đáng kể** | **Chính** |
| **Trôi tempo** | Có thật nhưng nhỏ (điều chỉnh trung vị 0.56%) | Phụ |
| **Hiss / nhiễu** | **Không phát hiện được** trên material này (−49…−57 dB) | Không |

Ba lỗi này **trải khắp toàn bài**, không cục bộ — đúng phạm vi của V3. Vì vậy nguồn lực dồn sang **V3, cụ thể là bước mở rộng dải tần**, là bước duy nhất tạo ra khác biệt nghe được rõ ràng.

**V2 không bị bỏ.** Nó vẫn trong pipeline, vẫn bật được (`do_stage_a`), vẫn giữ artifact riêng (`v2/<tên>_v2.wav`) để đối chiếu. Nó chỉ không còn là nơi đầu tư thời gian.

---

## 4. VERSION 3 — `suno_restore` (phục hồi toàn bài)

Thứ tự ba bước là **cố định và có chủ đích** (`suno_restore/pipeline.py`):

1. **Tempo trước** — đây là bước duy nhất động vào **trục thời gian**, và được tính **một lần chung cho mọi stem**. Chạy sau sẽ kéo giãn lại chính những gì các bước khác vừa sửa.
2. **Denoise ở giữa** — phải trước mở rộng dải tần, vì model siêu phân giải **tổng hợp dải cao từ những gì nằm dưới ngưỡng cắt**; cho nó ăn nhiễu thì nó nhân bản nhiễu lên dải cao.
3. **Bandwidth cuối** — là bước duy nhất đổi sample rate (Apollo xuất 44.1 kHz).

---

### 4.1 Bước 1 — Sửa tempo (`tempo.py`)

**Model:** **Beat This!** (`beat_this.inference.Audio2Beats`, checkpoint `"final0"`).

#### Phát hiện nhịp — một lần, dùng chung

Beat tracking chạy trên **một stem tham chiếu duy nhất**, theo thứ tự ưu tiên `drum → bass → instrumental → guitar → piano` (nhạc cụ gõ cho onset rõ nhất). Nếu stem đó không cho đủ nhịp thì thử ứng viên tiếp theo.

**Warp phải dùng chung cho mọi stem.** Nếu beat-track riêng từng stem, mỗi stem có một warp riêng và chúng **trôi lệch nhau** — mất luôn ý nghĩa của việc tách stem.

`target_interval = median(diff(beat_times))` → lưới nhịp đều để căn về.

#### Xây bản đồ warp (`warp_anchors`)

Sinh ra hai mảng **tăng nghiêm ngặt**, cùng độ dài: vị trí mẫu ở **nguồn** ↔ vị trí mẫu ở **đầu ra**. Vì cả hai đều tăng nghiêm ngặt nên ánh xạ giữa chúng **liên tục và khả nghịch** — đây chính là tính chất cho phép warp chạy **một lượt duy nhất**, và một lượt duy nhất thì **không có mối nối nào**.

Hai bảo vệ:
- `MIN_CORRECTION = 0.005` — lệch dưới 0.5% thì bỏ qua. Mọi phép tái tổng hợp đều tốn một chút nhòe; không nên tiêu nó để sửa sai số làm tròn.
- `MAX_RATE_DEVIATION = (0.80, 1.25)` — lệch quá ngưỡng này là **lỗi beat-tracking chứ không phải trôi tempo**; warp theo nó gây hại nhiều hơn cái nó định sửa.

**Xử lý nhịp bị mất:** Beat This! bỏ nhịp ở chỗ stem tham chiếu im lặng — trên bộ tham chiếu nó tìm được 188 nhịp trong khi 68 BPM đều đặn phải cho ~333, để lại khoảng trống tới 28 giây. Nên mỗi khoảng được **đo theo số nhịp nguyên** (`beats_spanned = round(length / target_samples)`) trước khi sửa. Ép mọi khoảng thành một nhịp làm bài **co lại còn 79% độ dài**.

Phần trước nhịp đầu tiên và sau nhịp cuối nằm ngoài lưới → giữ nguyên tốc độ 1.0.

#### Nội suy — PCHIP

Bản đồ được nội suy bằng **PCHIP** (monotone cubic) thay vì đường thẳng qua các mốc. Cả hai đặt nhịp **giống hệt nhau** (chúng trùng tại mọi mốc), nhưng bản tuyến tính có **gãy đạo hàm tại mỗi mốc** — tức là **tốc độ phát thay đổi đột ngột ở mỗi nhịp**. PCHIP liên tục cấp C1 và bảo toàn tính đơn điệu → tốc độ *lướt* mượt giữa các khoảng, và bản đồ vẫn không thể gập ngược.

#### Phase vocoder tốc độ biến thiên — một lượt

`N_FFT = 2048`, `HOP_LENGTH = 512`.

- STFT một lần trên **toàn bộ** tín hiệu.
- Mỗi khung đầu ra cách nhau `HOP_LENGTH` mẫu; nó đọc từ vị trí nguồn mà bản đồ chỉ tới (chỉ số khung **phân số**, nội suy biên độ giữa hai khung kề).
- Pha được cộng dồn bằng **lượng tiến pha thật trên một hop** (`expected + độ lệch đã wrap`), cộng dồn **một lần duy nhất trên cả bài** — bộ cộng dồn không bao giờ khởi động lại.

**Identity phase locking (Laroche–Dolson).** Phase vocoder cho mỗi bin tiến pha độc lập, nên các bin cùng mô tả một hoạ âm sẽ lệch pha nhau — hoạ âm vẫn đúng tần số nhưng **mất hình dạng sóng**, tạo ra tiếng "rỗng, óc ách" (phasiness) đặc trưng. Kỹ thuật này gán cho mỗi bin **lượng xoay pha của đỉnh phổ chi phối vùng lân cận nó**, giữ các bin của một hoạ âm xoay cùng nhau.

**Một lượng xoay pha dùng chung cho mọi kênh.** Lượng xoay được tính **một lần từ tín hiệu downmix (mono)** rồi áp cho tất cả các kênh. Cách làm thông thường — chạy vocoder độc lập cho L và R — **âm thầm làm rộng ra hoặc bóp hẹp ảnh stereo**, vì quan hệ pha giữa L và R *chính là* thứ tạo nên stereo, và hai bộ cộng dồn độc lập không bảo toàn nó.

#### Vì sao kiến trúc cũ bị thay (đây là lỗi vừa sửa)

Bản trước **cắt trục thời gian tại mỗi nhịp**, gọi `librosa.effects.time_stretch` **riêng cho từng mảnh**, rồi dán lại bằng crossfade tuyến tính 5 ms. Đo trên **sóng sin 440 Hz thuần** — tín hiệu có đường bao phẳng theo định nghĩa, nên mọi chỗ lõm đều là do xử lý tạo ra:

```
join  0 t=0.448s  min = −7.05 dB
join  3 t=1.778s  min = −5.38 dB
join  6 t=3.111s  min = −7.61 dB
Tổng: 78 sự kiện lõm > 1 dB, sâu nhất −9.2 dB
```

Ba nguyên nhân độc lập, đều đã xác nhận bằng đo đạc:

1. **`time_stretch` làm suy giảm phần đuôi mỗi mảnh** — đo được **−13.0 dB** ở vài ms cuối tại rate 0.99. Đó là dốc overlap-add của ISTFT, trải dài cả một cửa sổ 2048 mẫu; crossfade 5 ms không thể che nổi.
2. **Pha khởi động lại ở mỗi mảnh.** Mỗi lần gọi là một phase vocoder riêng → pha tại biên là tuỳ ý. Crossfade hai bản sao của cùng một hoạ âm ở pha khác nhau thì **triệt tiêu chứ không cộng**.
3. **Mỗi mối nối xoá mất 5 ms nhạc.** Mảnh có kéo giãn được cộng bù 5 ms; mảnh bị `MIN_CORRECTION` ép về rate 1.0 thì **không được bù** → mất hẳn 5 ms.

Trên file thật, lỗi này rơi **mỗi nhịp một lần**. Các điểm gián đoạn ở file ra mà file vào không có đều **lượng tử theo nhịp** (khoảng cách 0.45 / 0.86 / 1.12 / 1.82 / 3.41 s trên nền nhịp ~0.45 s).

**Không có crossfade nào chữa được, vì crossfade chính là nguyên nhân.** Nên bây giờ **không còn mảnh và không còn mối nối**.

Kết quả đo lại, cùng phép đo:

| Chỉ số | Cũ | Mới |
|---|---|---|
| Số sự kiện lõm đường bao | 78 | **0** |
| Lõm sâu nhất | −9.2 dB | **−0.31 dB** |
| Ảnh stereo (S/M) | — | **không đổi, 0.00 dB** |

Chạy toàn phần trên file stereo 185 s thật (với warp *khắc nghiệt hơn* thực tế — 5.5% trung vị so với ~1%): **6.9 s**, số bước nhảy mẫu > 0.05 **giảm** 10309 → 8979, ảnh stereo giữ trong 0.4 dB, và **tự tương quan sai số đường bao tại chu kỳ nhịp = +0.027** — dấu hiệu artifact khoá theo nhịp đã biến mất hoàn toàn.

---

### 4.2 Bước 2 — Khử ồn (`denoise.py`)

**Model:** **Mel-Band-RoFormer Denoise** — checkpoint `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`, chạy qua thư viện **`audio-separator`**.

Đây là model **music source separation**, xuất ra một track "dry" (đã sạch) và phần nhiễu nó tách ra. Nó khử hiss, crackle và "AI shimmer".

**Vì sao không dùng DeepFilterNet** dù nó điểm cao về noise suppression: nó tự mô tả là *"Speech Enhancement Framework"*. Trên bass, trống hay piano nó **triệt tiêu chính nội dung nhạc** mà nó không nhận ra là tiếng nói.

**Checkpoint ~900 MB** → phải cache; caller xử lý nhiều thư mục nên nạp một lần (`load_separator`).

#### Giữ ảnh stereo qua một model có thể chỉ xuất mono

Đây là chỗ dễ mất stereo nhất. `stereo_mode` có ba chế độ:

| Chế độ | Hành vi |
|---|---|
| `joint` | Tin model, chạy một lượt |
| `mid_side` | Luôn tách mid/side, chạy model **hai lượt** |
| `auto` *(mặc định)* | Chạy bình thường; **chỉ** trả giá cho lượt mid/side nếu thấy model ghi ra ít kênh hơn đầu vào |

`auto` đúng trong cả hai trường hợp và **không tốn gì khi model hoạt động tử tế**. Khi `auto` vừa *phát hiện* model là mono-only, nó **làm lại stem đó ngay** theo đường mid/side thay vì giao ra một stem đã vứt mất kênh side.

Mid/side: `mid = (L+R)/2`, `side = (L−R)/2`, khử ồn **độc lập từng cái**, rồi dựng lại `L = mid+side`, `R = mid−side`.

#### Chuẩn hoá gain xác định (vừa bổ sung)

`audio-separator` **tự chuẩn hoá bất kỳ tín hiệu nào có đỉnh vượt 0.9** (`normalization_threshold = 0.9`) và **không báo lại hệ số gain**. Hệ quả: hai lượt chạy trên cùng một stem có thể bị scale khác nhau. Điều này vô hại ở mono nhưng **phá hoại ở mid/side** — mid thường vượt ngưỡng còn side thì không, nên hai cái trở về ở hai thang khác nhau và **ảnh stereo bị dựng lại sai độ rộng**.

Khắc phục: nhân trước một hệ số để đỉnh luôn ≤ 0.5 (model không bao giờ chuẩn hoá), rồi chia lại đúng hệ số đã biết sau khi xong. Nhờ đó **gain của mọi lượt là như nhau và biết trước**.

#### Báo cáo

`residual_db` = mức của phần đã bị lấy đi, so với tín hiệu:

| Mức | Diễn giải |
|---|---|
| `−inf` | Không lấy đi gì |
| `< −40 dB` | Phần bị lấy đi không nghe được — stem này vốn đã sạch |
| `< −20 dB` | Lấy đi một lớp nhiễu nhỏ |
| `≥ −20 dB` | **Lấy đi đáng kể — cần kiểm tra xem có mất nội dung nhạc không** |

> Trên material tham chiếu, bước này đo được **−49 đến −57 dB**, tức là **về cơ bản không làm gì**. Nó vẫn có sẵn nhưng **không nên báo cáo như một lợi ích mà nó không mang lại**.

---

### 4.3 Bước 3 — Mở rộng dải tần (`bandwidth.py`) — **bước chính**

**Model:** **Apollo** (`JusperLee/Apollo`, checkpoint `pytorch_model.bin` tải từ HuggingFace Hub).
**Kiến trúc:** `sr=44100, win=20, feature_dim=256, layer=6` — đúng cấu hình mà script inference của Apollo khởi tạo.

Apollo được huấn luyện trên **artifact của codec MP3** và đánh giá trên dữ liệu **stem đã tách** (MUSDB18-HQ, MoisesDB) — đúng loại material ở đây.

#### Vì sao không dùng AudioSR

1. Nó cấp phát ~**7.2 GB**, **không vừa card 8 GB** mà hệ thống này chạy.
2. Chính README của nó cảnh báo: huấn luyện trên dữ liệu low-pass **tổng hợp** khiến nó yếu ở đúng loại ngưỡng cắt hình dạng-MP3 mà các stem này có.

#### Vì sao không dùng `inference.py` của Apollo

Apollo không có gói pip; `look2hear` được import từ một bản clone (`vendor/Apollo`). Script đi kèm **cố tình không dùng**, vì ba lỗi:

1. Nó gọi `from_pretrain("JusperLee/Apollo")`, nhưng `from_pretrain` chạy `torch.load()` trên chuỗi đó. Đây là **đường dẫn checkpoint cục bộ, không phải repo id** → script **hỏng thẳng** với `FileNotFoundError`.
2. Nó **hardcode `.cuda()`** ở hai chỗ, không có đường lui về CPU.
3. **Loader của nó không resample.** Nó đẩy thẳng sample rate của file vào một model xây cho 44.1 kHz rồi ghi ra ở 44.1 kHz → đầu vào 48 kHz **trả về bị dịch cao độ và tốc độ ~8.8%** (48000/44100).

Làm ở mức thư viện tránh cả ba: resample về 44.1 kHz khi vào, resample về lại khi ra, thiết bị chọn lúc chạy.

#### Chunking và crossfade

Apollo là model chuỗi; cả một stem một lúc thì vượt VRAM. Nên chạy theo khối:

- `bandwidth_chunk_s = 10.0`, `bandwidth_overlap_s = 2.0` (`step = chunk − overlap = 8 s`).

**Cửa sổ `sin²`** (`chunk_window`). Chọn hình dạng này vì hai lý do:

- **Bù trừ chính xác:** `sin² + cos² = 1`, nên hai cửa sổ chồng lên nhau tạo thành **phân hoạch đơn vị (partition of unity)** đúng bằng 1. Điều này quan trọng: dải thấp gần như **giống hệt nhau** ở hai khối kề, và bất kỳ hàm cửa sổ nào **không** cộng lại bằng 1 sẽ **nâng hoặc cắt** dải thấp tại mỗi mối nối.
- **Trơn:** dốc tuyến tính có **gãy đạo hàm** ở hai đầu; `sin²` chạm 0 và 1 với **đạo hàm bằng 0**, nên mối nối không tạo nếp gãy để phổ tán năng lượng ra xung quanh.

**Vì sao overlap là 2.0 s chứ không phải 0.5 s:** fade ngắn để lại một **dao động mức trong dải cao được tổng hợp, bám đúng theo bước nhảy khối**. Cần lưu ý: cửa sổ trơn **không** chữa được nửa còn lại của vấn đề — ở chỗ hai khối **bất đồng** (đúng bản chất của dải cao được tổng hợp độc lập), trung bình có trọng số của nội dung không tương quan **mất công suất**, tệ nhất 3 dB. Đo thực tế: fade dài **chỉ có lợi khi hai khối phần lớn đồng thuận** — là trường hợp thực tế khi bước này được bật đúng chỗ. Cả hai hiệu ứng đều dưới ~1 dB.

**Không fade ở đầu và cuối file:** ở đó không có khối kề để fade vào, nên fade chỉ làm **tối đi giây đầu và giây cuối**; và vì overlap-add chuẩn hoá theo tổng trọng số, khối cuối fade về 0 sẽ dẫn tới **chia cho gần như không có gì**.

#### Gia cố `process_chunked` (vừa bổ sung)

- **Các khoảng khối và cờ fade được quyết định trước khi xử lý**, dựa trên khối nào thực sự kề khối nào. Trước đây câu hỏi "đây có phải khối cuối không?" được hỏi từ một vị trí **chưa thể biết** liệu có khối tiếp theo được lấy hay không — và một cửa sổ fade-out mà không có gì để fade vào chính là **một lỗ hổng trong vùng phủ**.
- **Model trả về ngắn hơn kỳ vọng** thì phần đuôi được **lấp bằng chính tín hiệu nguồn**. Trước đây phần đó không được khối nào gánh, và phép chuẩn hoá sẽ **chia gần-im-lặng cho gần-không** (`np.maximum(weights, 1e-8)`) → **một tiếng glitch rất to**. Không mở rộng dải tần là một kiểu thất bại tốt hơn nhiều so với khuếch đại nhiễu.

---

## 5. Đo đạc (`quality.py`, `metrics.py`)

**Nguyên tắc: các chỉ số mô tả bước đó đã làm gì; chúng không bao giờ quyết định bước đó có chạy hay không.**

### 5.1 `metrics.py` — chỉ số theo bước

- **`spectral_cliff_hz`** — tần số cao nhất nơi phổ rơi ≥ 25 dB trong vòng 1500 Hz. Dùng **percentile 95** thay vì trung bình (khoảng lặng giữa các nốt kéo trung bình xuống và làm nhoè một ngưỡng cắt vốn rất sắc). Đo **tương đối theo độ dốc**, không theo đỉnh phổ — đo theo đỉnh sẽ đọc một stem bass là "cắt ở 400 Hz" chỉ vì bass vốn không có treble.
  `floor_hz = 2000` là giới hạn thật: ngưỡng cắt **thấp hơn** mức này **không thể tìm ra**, vì đến lúc bắt đầu dò thì tín hiệu đã ở sàn nhiễu.
- **`energy_above_hz`** — % năng lượng trên 13 kHz.
- **`residual_level_db`** — mức phần denoise đã lấy đi.
- **`tempo_stability`** — độ ổn định lưới nhịp.

### 5.2 `quality.py` — so sánh gốc ↔ đã xử lý

| Nhóm | Nội dung |
|---|---|
| `TimingMetrics` | Căn chỉnh, độ trễ, thay đổi độ dài |
| `SpectralMetrics` | Ngưỡng cắt, cân bằng dải |
| `TransientMetrics` | Độ sắc của onset |
| `DynamicMetrics` | **LUFS tích hợp (ITU-R BS.1770 / EBU R128)**, loudness range |
| `StereoMetrics` | Tương quan, độ rộng ảnh stereo |
| `ArtifactMetrics` | Kết dính dải cao, **độ sâu điều biến dải cao**, bước nhảy mẫu lớn nhất |

**`hf_modulation_increase_db` là chỉ số then chốt cho artifact mối nối khối.** Nhạc tự nó đã có biến thiên chậm ở mức dải cao, nên độ sâu điều biến của tín hiệu đã xử lý **không nói lên gì nếu đứng một mình** — chỉ phần **tăng thêm so với nguồn** mới quy được cho khâu xử lý.

**`hf_envelope_correlation`** trả lời câu hỏi mà `hf_coherence` không trả lời được: với một nguồn thực sự bị giới hạn băng thông, **không có gì ở trên ngưỡng cắt để mà kết dính với**, nên mọi phép mở rộng đều gần 0 và bản tốt không phân biệt được với nhiễu. Câu hỏi trả lời được là: **dải cao có lên xuống theo nhạc không?** Nội dung dẫn xuất từ nguồn thì có; nội dung bịa ra thì không.

Công cụ dòng lệnh: `python scripts/ab_quality.py <gốc> <đã_phục_hồi>`

---

## 6. Quản lý GPU

Model của V2 (BS-RoFormer, MuScriptor, AMT) và của V3 (Mel-Roformer ~900 MB, Apollo) **không được cùng thường trú** trên card nhỏ. `midi_repair.unload_models()` chạy trong khối `finally` ở ranh giới V2 → V3, **kể cả khi V2 báo lỗi**.

Trong V3, mỗi stem sau khi ghi xong sẽ được **xoá khỏi bộ nhớ ngay** (`stems.pop(name)`) — giữ mọi stem đã phục hồi tốn ~110 MB mỗi cái ở độ dài đầy đủ và không có bước nào phía sau cần đến.

---

## 7. Trạng thái hiện tại

### 7.1 Đã hoàn thành và đã kiểm chứng

| Thành phần | Trạng thái |
|---|---|
| Giải mã ffmpeg, giữ sample rate/kênh gốc | ✅ Hoạt động |
| Hợp đồng số kênh (chống sập stereo) | ✅ Có chốt chặn sau mỗi bước |
| V2 — sửa cục bộ + rollback theo rủi ro + chỉ áp delta | ✅ Hoạt động |
| V3 Bước 1 — Tempo warp liên tục một lượt | ✅ **Vừa viết lại — artifact mối nối đã hết** |
| V3 Bước 2 — Denoise Mel-Roformer + mid/side | ✅ Hoạt động (gain đã xác định) |
| V3 Bước 3 — Apollo + overlap-add | ✅ Hoạt động (overlap-add đã gia cố) |
| Đo đạc `quality.py` | ✅ Đầy đủ |
| Bộ test | ✅ **62/62 pass**; `ruff check` sạch |

### 7.2 Kết quả sửa gần nhất (11/08/2026)

Nguyên nhân gốc của hiện tượng "âm thanh chưa mượt, còn ngắt quãng nhẹ" đã được xác định là **bước 1 — tempo**, không phải denoise hay bandwidth. Đã thay kiến trúc cắt-mảnh-rồi-dán bằng **warp liên tục một lượt**. Xem §4.1.

### 7.3 Vấn đề còn mở

**Ảnh stereo của bản xuất bị hẹp lại ~6 dB.**

| Chỉ số | File vào | File ra |
|---|---|---|
| S/M (trung vị) | −6.1 dB | **−12.0 dB** |
| Tương quan L/R | 0.83 | **0.99** |

Một phần là **dự kiến được** — hiss vốn không tương quan, nên khử ồn tất yếu làm hẹp lại. Nhưng **6 dB là nhiều**. Bản sửa tempo (xoay pha dùng chung) và bản sửa gain denoise đều đã loại bỏ hai nguyên nhân khả dĩ. **Cần nghe lại và đo lại sau khi chạy pipeline mới**; nếu vẫn hẹp thì đây là mục tiếp theo cần truy.

**Đã kiểm tra và loại trừ:** ban đầu có 39 "khoảng lặng cứng" bị nghi là dropout trong file nguồn. Kiểm tra kỹ cho thấy đó là **điểm triệt tiêu của kênh mid nằm trong những đoạn vốn đã im lặng** (đỉnh của vùng lân cận = 0.000) — **không phải ngắt quãng**. Một bước "vá dropout" ở đây sẽ là **phá hoại**, nên đã bỏ ý định đó.

### 7.4 Chưa xác nhận

Chạy end-to-end thật với đầy đủ checkpoint nhiều GB trên **GPU triển khai** và **ghi lại đỉnh VRAM tại ranh giới V2 → V3** — theo `docs/design/2026-08-10-merged-pipeline.md`, việc này vẫn cần làm trước khi phát hành.

---

## 8. Cấu hình

Mọi tuỳ chọn đọc từ biến môi trường tên `RESTORE_<TÊN_TRƯỜNG>`. **Không có file `.env` nào được nạp** — đọc thẳng từ môi trường tiến trình.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `RESTORE_OUTPUT_SAMPLE_RATE` | `0` | `0` = đi theo nguồn |
| `RESTORE_STEREO_MODE` | `auto` | `auto` / `mid_side` / `joint` |
| `RESTORE_ENFORCE_CHANNEL_COUNT` | `true` | Chốt chặn số kênh |
| `RESTORE_BANDWIDTH_CHUNK_S` | `10.0` | Độ dài khối Apollo |
| `RESTORE_BANDWIDTH_OVERLAP_S` | `2.0` | Độ chồng lấn giữa hai khối |

---

## 9. Tóm tắt model theo từng bước

| Pipeline | Bước | Model / thư viện | Nguồn |
|---|---|---|---|
| Chung | Giải mã | ffmpeg / ffprobe | Hệ thống |
| Chung | Resample | soxr | pip |
| **V2** | Tách stem | **BS-RoFormer** | `bs-roformer-infer` |
| **V2** | Audio → MIDI | **MuScriptor** (large/medium) | `muscriptor` |
| **V2** | Phân tích key / DTW | music21, dtw-python, librosa | pip |
| **V2** | Tái tạo MIDI | **stanford-crfm/music-small-800k** | HF + `anticipation` |
| **V2** | Render donor | librosa `pitch_shift` / `time_stretch` | pip |
| **V3** | Bước 1 — Beat tracking | **Beat This!** (`final0`) | `beat_this` |
| **V3** | Bước 1 — Warp | Phase vocoder biến thiên tự viết + PCHIP | `suno_restore/tempo.py` |
| **V3** | Bước 2 — Denoise | **Mel-Band-RoFormer Denoise** (SDR 27.9959) | `audio-separator` |
| **V3** | Bước 3 — Bandwidth | **Apollo** | `JusperLee/Apollo` (HF + clone) |
| Đo đạc | LUFS / phổ / stereo | Tự viết theo ITU-R BS.1770 | `suno_restore/quality.py` |
