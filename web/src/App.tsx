import React, { useMemo, useState } from "react";

type GenerateParams = {
  paint_thickness: number;
  messiness: number;
  text_wobble: number;
  seed: number;
  blur_sigma: number;
  blur_mix: number;
  shadow_opacity: number;
  shadow_sigma: number;
  shadow_dx: number;
  shadow_dy: number;
};

const DEFAULTS: GenerateParams = {
  paint_thickness: 10,
  messiness: 0.01,
  text_wobble: 0.1,
  seed: 1500,
  blur_sigma: 1.5,
  blur_mix: 0.5,
  shadow_opacity: 0.18,
  shadow_sigma: 6,
  shadow_dx: 1,
  shadow_dy: 1,
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState<string>("");
  const [params, setParams] = useState<GenerateParams>(DEFAULTS);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);

  const canSubmit = file && text.trim().length > 0 && !isLoading;

  const previewUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!file) {
      setError("Please choose an image.");
      return;
    }
    if (text.trim().length === 0) {
      setError("Please enter text.");
      return;
    }

    setIsLoading(true);
    try {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("text", text);
      fd.append("paint_thickness", String(params.paint_thickness));
      fd.append("messiness", String(params.messiness));
      fd.append("text_wobble", String(params.text_wobble));
      fd.append("seed", String(params.seed));
      fd.append("blur_sigma", String(params.blur_sigma));
      fd.append("blur_mix", String(params.blur_mix));
      fd.append("shadow_opacity", String(params.shadow_opacity));
      fd.append("shadow_sigma", String(params.shadow_sigma));
      fd.append("shadow_dx", String(params.shadow_dx));
      fd.append("shadow_dy", String(params.shadow_dy));

      const res = await fetch("/generate", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `Request failed: ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      // Clean up old blob url.
      if (resultUrl) URL.revokeObjectURL(resultUrl);
      setResultUrl(url);
    } catch (err: any) {
      setError(err?.message ?? "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Whiteout</h1>
          <p className="sub">
            ▸ Upload photo ▸ Add text ▸ Get lo-fi painted patch
          </p>
        </div>
        {/* <a
          className="link"
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          rel="noreferrer"
        >
          API DOCS
        </a> */}
      </header>

      <div className="grid">
        <form className="card" onSubmit={onSubmit}>
          <h2>Input</h2>

          <label className="label">
            Image
            <input
              className="input"
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>

          <label className="label">
            Text
            <textarea
              className="textarea"
              placeholder="Type exactly what you want…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
            />
          </label>

          <details className="details">
            <summary>Advanced</summary>
            <div className="advanced">
              <NumberField
                label="paint_thickness"
                value={params.paint_thickness}
                min={1}
                max={200}
                step={1}
                onChange={(v) =>
                  setParams((p) => ({ ...p, paint_thickness: v }))
                }
              />
              <NumberField
                label="messiness"
                value={params.messiness}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => setParams((p) => ({ ...p, messiness: v }))}
              />
              <NumberField
                label="text_wobble"
                value={params.text_wobble}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => setParams((p) => ({ ...p, text_wobble: v }))}
              />
              <NumberField
                label="seed"
                value={params.seed}
                min={0}
                max={999999999}
                step={1}
                onChange={(v) => setParams((p) => ({ ...p, seed: v }))}
              />
              <NumberField
                label="blur_sigma"
                value={params.blur_sigma}
                min={0}
                max={10}
                step={0.1}
                onChange={(v) => setParams((p) => ({ ...p, blur_sigma: v }))}
              />
              <NumberField
                label="blur_mix"
                value={params.blur_mix}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => setParams((p) => ({ ...p, blur_mix: v }))}
              />
              <hr className="sep" />
              <NumberField
                label="shadow_opacity"
                value={params.shadow_opacity}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) =>
                  setParams((p) => ({ ...p, shadow_opacity: v }))
                }
              />
              <NumberField
                label="shadow_sigma"
                value={params.shadow_sigma}
                min={0}
                max={50}
                step={0.1}
                onChange={(v) => setParams((p) => ({ ...p, shadow_sigma: v }))}
              />
              <NumberField
                label="shadow_dx"
                value={params.shadow_dx}
                min={-50}
                max={50}
                step={1}
                onChange={(v) => setParams((p) => ({ ...p, shadow_dx: v }))}
              />
              <NumberField
                label="shadow_dy"
                value={params.shadow_dy}
                min={-50}
                max={50}
                step={1}
                onChange={(v) => setParams((p) => ({ ...p, shadow_dy: v }))}
              />
            </div>
          </details>

          <div className="row">
            <button className="button" type="submit" disabled={!canSubmit}>
              {isLoading ? "Generating…" : "Generate"}
            </button>
            <button
              className="button ghost"
              type="button"
              onClick={() => {
                setParams(DEFAULTS);
              }}
            >
              Reset defaults
            </button>
          </div>

          {error && <div className="error">{error}</div>}
        </form>

        <div className="card">
          <h2>Preview</h2>
          <div className="previewGrid">
            <div>
              <div className="previewTitle">Original</div>
              <div className="previewPane">
                {previewUrl ? (
                  <img src={previewUrl} alt="original" />
                ) : (
                  <div className="muted">No image yet</div>
                )}
              </div>
            </div>
            <div>
              <div className="previewTitle">Result</div>
              <div className="previewPane">
                {resultUrl ? (
                  <img src={resultUrl} alt="result" />
                ) : (
                  <div className="muted">Generate to see output</div>
                )}
              </div>
              {resultUrl && (
                <a className="download" href={resultUrl} download="out.png">
                  Download PNG
                </a>
              )}
            </div>
          </div>
        </div>
      </div>

      <footer className="footer">
        Keep <code>seed</code> the same for deterministic output. GAME ON!
      </footer>
    </div>
  );
}

function NumberField(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="label small">
      {props.label}
      <input
        className="input"
        type="number"
        value={Number.isFinite(props.value) ? props.value : ""}
        min={props.min}
        max={props.max}
        step={props.step}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}
