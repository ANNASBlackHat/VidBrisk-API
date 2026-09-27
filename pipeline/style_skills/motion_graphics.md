# Motion Graphics Concepting Skill

This skill defines the rules for selecting and formatting motion-graphic components when:
1. A beat was designated as a non-narrative beat during initial structuring (`stat`, `kinetic`, etc.), OR
2. A narrative beat had `footage_status == INADEQUATE` and fell back to a motion-graphic `takeover`.

---

## Registered Motion Components & Selection Rules

### 1. `DataAnimations/StatCard` (`beat_type: "stat"`)
- **When to choose**: The beat contains prominent numbers, percentages, financial amounts, metrics, or quantified comparisons (e.g. "$25.4B", "276 teeth", "43% of turtles", "36,000 feet").
- **Component ID**: `DataAnimations/StatCard`
- **Recipe**: `stat_over_footage` if background footage is present; otherwise pure `takeover`.
- **Props**:
  - `primary_value`: String formatted metric (e.g. "$25.4B", "276", "43%", "36,000 FT").
  - `kicker`: Short 2–4 word uppercase label (e.g. "TEETH AT A TIME", "TOTAL EXPEDITION COST", "MAX DEPTH").
  - `visual_type`: `"chart"` (for trends/money), `"ring"` (for percentages/fractions), or `"bar"` (for counts/comparisons).
  - `subtext`: 1 brief contextual sentence explaining the significance.
  - `display_mode`: `"takeover"` (for footage fallbacks) or `"overlay"`.

### 2. `TextAnimations/KineticText` (`beat_type: "kinetic"`)
- **When to choose**: The beat is a high-energy statement, punchy hook, or rhythmic phrase where synchronized word-by-word reveal delivers maximum impact.
- **Component ID**: `TextAnimations/KineticText`
- **Props**:
  - `text`: Narration phrase for dynamic reveal.
  - `mode`: `"reveal"` | `"karaoke"`.
  - `display_mode`: `"takeover"` | `"overlay"`.

### 3. `TextAnimations/Typewriter` (`beat_type: "typewriter"`)
- **When to choose**: Terminal readout, historical log entry, mission record, or investigative reveal (e.g., "MISTAKE #1", "LOG: 04:12 MARIANA BASIN").
- **Component ID**: `TextAnimations/Typewriter`
- **Props**:
  - `text`: Exact text to type out character by character.
  - `display_mode`: `"takeover"`.

### 4. `TextAnimations/QuoteCard` (`beat_type: "quote"` or `"abstract"`)
- **When to choose**: Philosophical statements, attributed spoken quotes, or core thesis takeaways without numbers.
- **Component ID**: `TextAnimations/QuoteCard`
- **Recipe**: `quote_over_footage` if background footage exists; otherwise `takeover`.
- **Props**:
  - `quote`: The core quoted sentence or thought.
  - `emphasis`: 2–5 words within the quote highlighted with accent color.
  - `author`: Attributed speaker or context (e.g., "DR. ROBERT BALLARD" or "EXPEDITION LOG").
  - `display_mode`: `"takeover"` | `"overlay"`.

### 5. `ListAnimations/SwipeDeck` (`beat_type: "swipe_deck"`)
- **When to choose**: Sequential 3–5 takeaways, key steps, enumerated points, or layered facts.
- **Component ID**: `ListAnimations/SwipeDeck`
- **Props**:
  - `title`: Short uppercase category title (e.g. "THREE CRITICAL MISTAKES", "DISCOVERY TIMELINE").
  - `items`: Array of 3–5 short, punchy bullet points (each under 12 words).
  - `display_mode`: `"takeover"`.

### 6. `ListAnimations/ChatBubbles` (`beat_type: "chat_bubbles"`)
- **When to choose**: Conversational exchanges, internet discourse, reports flooding social media, or dialogue between two perspectives.
- **Component ID**: `ListAnimations/ChatBubbles`
- **Props**:
  - `title`: Header label (e.g. "INTERNET REPORTS", "COMMUNICATION LOG").
  - `messages`: Array of 2–4 message objects: `[{"text": "...", "sender": "user"|"system"}]`.
  - `display_mode`: `"takeover"`.

### 7. `Layouts/SplitScreen` (`beat_type: "split_screen"`)
- **When to choose**: Direct contrast or dual comparison (e.g., Megalodon vs Great White, Depth vs Pressure, Theory vs Reality).
- **Component ID**: `Layouts/SplitScreen`
- **Recipe**: `split_screen`.
- **Props**:
  - `leftTitle`: Left entity title.
  - `leftContent`: Left entity description or metric.
  - `rightTitle`: Right entity title.
  - `rightContent`: Right entity description or metric.
  - `display_mode`: `"takeover"`.

### 8. `GeoAnimations/MapExplainer` (`beat_type: "map_route"`)
- **When to choose**: Geographic navigation, expedition path between two locations, or landmark location callout.
- **Component ID**: `GeoAnimations/MapExplainer`
- **Props**:
  - `origin`: Starting location / site name.
  - `destination`: Destination location / site name.
  - `mode`: `"route"` (between 2 points) or `"pin"` (single landmark).
  - `title`: Header label (e.g. "MARIANA EXPEDITION TRAJECTORY").
  - `display_mode`: `"takeover"`.

### 9. `AudioAnimations/AudioWaveform` (`beat_type: "audio_waveform"`)
- **When to choose**: Radio comms, sonar recordings, sonar pings, dispatch transmissions, or podcast quotes.
- **Component ID**: `AudioAnimations/AudioWaveform`
- **Props**:
  - `speaker`: Callsign or speaker name.
  - `quote`: Key spoken transmission phrase.
  - `title`: Category header (e.g. "SONAR TRANSMISSION LOG").
  - `display_mode`: `"takeover"` | `"overlay"`.

---

## Fallback Adaptation Strategy (When footage is INADEQUATE)

When a narrative beat falls back to motion graphics because no footage met the quality bar:
1. **Always default `display_mode` to `"takeover"`**: There is no usable background footage, so the graphic must fill the canvas cleanly.
2. **Component Prioritization**:
   - Contains numbers / metrics / dates → **`DataAnimations/StatCard`**
   - Contains lists / steps / points → **`ListAnimations/SwipeDeck`**
   - Dramatic conclusion / thesis → **`TextAnimations/KineticText`** or **`TextAnimations/Typewriter`**
   - Quote or reflective thought → **`TextAnimations/QuoteCard`**
   - Comparison / vs → **`Layouts/SplitScreen`**
   - Geographic coordinates / sites → **`GeoAnimations/MapExplainer`**
   - Otherwise → **`TextAnimations/KineticText`** (karaoke or reveal) as a dynamic full-screen title.
