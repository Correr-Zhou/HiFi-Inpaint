import argparse
import shutil
import subprocess
import warnings
from pathlib import Path

try:
    import pypdfium2 as pdfium
    from PIL import Image, ImageChops
except Exception:
    pdfium = None
    Image = None
    ImageChops = None

if Image is not None:
    Image.MAX_IMAGE_PIXELS = None
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def build_default_mappings(latex_dir: Path, output_dir: Path):
    return {
        latex_dir / "figure_files" / "teaser.pdf": output_dir / "teaser.jpg",
        latex_dir / "figure_files" / "qual_results.pdf": output_dir / "qual_results.jpg",
        latex_dir / "figure_files" / "hifi.pdf": output_dir / "pipeline.jpg",
    }


def trim_whitespace(image: "Image.Image", threshold: int):
    if Image is None or ImageChops is None:
        return image
    rgb_image = image.convert("RGB")
    bg = Image.new("RGB", rgb_image.size, (255, 255, 255))
    diff = ImageChops.difference(rgb_image, bg).convert("L")
    diff = diff.point(lambda x: 255 if x > threshold else 0)
    bbox = diff.getbbox()
    if bbox:
        return rgb_image.crop(bbox)
    return rgb_image


def render_with_pdfium(pdf_path: Path, dpi: int):
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[0]
    scale = dpi / 72
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil()
    page.close()
    pdf.close()
    return image


def convert_pdf(
    pdf_path: Path,
    output_path: Path,
    max_width: int,
    image_format: str,
    quality: int,
    dpi: int,
    trim: bool,
    trim_threshold: int,
    dry_run: bool,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pdfium and Image:
        if dry_run:
            print(f"pdfium {pdf_path} -> {output_path}")
            return
        image = render_with_pdfium(pdf_path, dpi)
        if trim:
            image = trim_whitespace(image, trim_threshold)
        if max_width and image.width > max_width:
            new_height = int(image.height * (max_width / image.width))
            image = image.resize((max_width, new_height), Image.LANCZOS)
        output_format = "JPEG" if image_format in {"jpeg", "jpg"} else "PNG"
        save_kwargs = {}
        if output_format == "JPEG":
            save_kwargs["quality"] = quality
            save_kwargs["subsampling"] = 0
            image = image.convert("RGB")
        image.save(output_path, format=output_format, **save_kwargs)
        return

    if shutil.which("sips") is None:
        raise SystemExit("sips is required but not found")

    command = [
        "sips",
        "-s",
        "format",
        image_format,
        "-Z",
        str(max_width),
        str(pdf_path),
        "--out",
        str(output_path),
    ]
    if image_format in {"jpeg", "jpg"}:
        command = [
            "sips",
            "-s",
            "format",
            "jpeg",
            "-s",
            "formatOptions",
            str(quality),
            "-Z",
            str(max_width),
            str(pdf_path),
            "--out",
            str(output_path),
        ]
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex-dir", default="../CVPR_2026___HiFi_Inpaint")
    parser.add_argument("--output-dir", default="../HiFi-Inpaint/static/images")
    parser.add_argument("--max-width", type=int, default=2600)
    parser.add_argument("--format", default="jpg", choices=["jpg", "jpeg", "png"])
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--trim", dest="trim", action="store_true", default=True)
    parser.add_argument("--no-trim", dest="trim", action="store_false")
    parser.add_argument("--trim-threshold", type=int, default=8)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    latex_dir = Path(args.latex_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    mappings = build_default_mappings(latex_dir, output_dir)

    if args.all:
        for pdf_path in latex_dir.glob("figure_files/*.pdf"):
            mappings[pdf_path] = output_dir / f"{pdf_path.stem}.{args.format}"

    missing = [path for path in mappings if not path.exists()]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing PDF files:\n{missing_list}")

    for pdf_path, output_path in mappings.items():
        convert_pdf(
            pdf_path,
            output_path,
            args.max_width,
            args.format,
            args.quality,
            args.dpi,
            args.trim,
            args.trim_threshold,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
