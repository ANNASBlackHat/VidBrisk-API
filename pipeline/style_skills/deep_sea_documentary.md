# Deep Sea / Cryptid Documentary — Visual Style

## Tone & Mood
- Default mood: `somber` or `tense`. Avoid `hopeful`, `triumphant`, upbeat, or slick.
- Investigative, restrained narration style. Let the footage breathe.

## Beat Type Preferences
- **Favor** `narrative` beats over motion-graphic takeovers.
- **Reserve** `stat` for: depth measurements, dates, specimen counts,
  percentages (e.g. "276 teeth at a time", "3.6 million years ago").
- **Avoid** `kinetic`, `chat_bubbles`, `swipe_deck` — too casual for
  investigative documentary tone.
- `typewriter` acceptable only for archival-style record reveals.
- `audio_waveform` only for radio/mission voice transmission quotes.

## Visual Intent Heuristics
- **Unfilmable subjects** (extinct animals, hypothetical events): name the
  closest real analog explicitly in `visual_intent`. Do not search for the
  literal subject; find what actually exists on camera.
  Example: "great white shark swimming in open water — analog for megalodon body shape"
- Favor: fog, wreckage, murky deep-sea water, archival maritime footage,
  muted/desaturated color, wide static shots over handheld energy.
- Avoid: bright studio lighting, fast-cut energy, colorful infographic styles,
  static stock-site landscape loops.
- **Physical evidence beats** (teeth, fossils, specimens): close-up hands-on
  shots with a scale reference object (ruler, pen, other tooth) in frame.
- **Historical events without footage**: reenactment framing or archival
  black-and-white photography.
- **Research / internet report beats**: screen-recording style — actual
  search results page, browser scroll, news headlines as meta-visual.

## Layout & Motion
- Stats: use `layout_recipe: "stat_over_footage"` and `display_mode: "overlay"`
  to preserve footage continuity. Use `display_mode: "takeover"` only for
  major chapter-marker moments.
- Chapter markers (e.g. "MISTAKE #1", "THE DISCOVERY"): `beat_type: "kinetic"`
  or `"typewriter"` on a black background, `display_mode: "takeover"`.
- CGI is appropriate when the claim requires showing something no real camera
  could ever capture (scale comparison between species, skeleton/x-ray reveal,
  extinct animal in a modern context).

## Dynamism
- Prefer higher-motion footage (crashing waves, active shark swimming, creature
  pursuing prey) over static seascape B-roll.
- Avoid looping 1-second stock clips; prefer footage with genuine scene arc
  (camera movement, lighting change, subject entering frame).
