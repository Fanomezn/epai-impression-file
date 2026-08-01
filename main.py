import asyncio
import base64
import io
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import cairosvg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import ImageFont
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfWriter

# -----------------------------------------------------------------------
# 1. INITIALISATION ET CONFIGURATION
# -----------------------------------------------------------------------
app = FastAPI(title="Web2Print Recto/Verso API", version="2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remplacez la clé en dur par :
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

FALLBACK_MODELS = [
    "gemini-2.5-flash",       # Stable and highly reliable default
    "gemini-flash-latest",    # Fallback alias
]

PRODUCT_DIMENSIONS: Dict[str, Dict[str, str]] = {
    "business_card": {
        "viewBox": "0 0 850 540",
        "aspect": "PAYSAGE (HORIZONTAL) - Ratio 85:54 (Carte de visite)",
    },
    "flyer_a5": {
        "viewBox": "0 0 1480 2100",
        "aspect": "PORTRAIT (VERTICAL) - Ratio 148:210 (Flyer A5 - Recto accroche, Verso détails/contact)",
    },
    "flyer_a6": {
        "viewBox": "0 0 1050 1480",
        "aspect": "PORTRAIT (VERTICAL) - Ratio 105:148 (Flyer A6 - Recto accroche, Verso détails/contact)",
    },
    "depliant": {
        "viewBox": "0 0 2000 2100",
        "aspect": "SQUARE / PAYSAGE - Ratio 200:210 (Dépliant - Volets structurés)",
    },
    "bookmark": {
        "viewBox": "0 0 500 2000",
        "aspect": "PORTRAIT TRÈS ÉTROIT - Ratio 50:200 (Marque-page tout en verticalité)",
    },
    "xbanner": {
        "viewBox": "0 0 1200 2000",
        "aspect": "PORTRAIT GRAND FORMAT - Ratio 60:100 (X-Banner / Roll-up - Tout en un : Logo en haut, Infos au milieu, Contact en bas)",
    },
}

VALID_PRINT_MODES = {"recto", "recto_verso"}

BACKGROUND_STYLE_LABELS: Dict[str, str] = {
    "automatique": "Laisse l'IA choisir librement le fond/l'ambiance le plus adapté au secteur d'activité.",
    "photographique_realiste": "Fond à dominante photographique réaliste, scène immersive ou texture riche évoquant une photo de haute qualité.",
    "photo réaliste": "Fond à dominante photographique réaliste, scène immersive ou texture riche évoquant une photo de haute qualité.",
    "photorealiste": "Fond à dominante photographique réaliste, scène immersive ou texture riche évoquant une photo de haute qualité.",
    "couleur_unie_degrade": "Fond en couleur unie ou en dégradé, sans élément photographique.",
    "lumieres_neons": "Ambiance lumineuse type néons / bokeh lumineux.",
    "nature_exterieur": "Ambiance nature / extérieur (végétation, plein air, lumière naturelle).",
    "urbain_ville": "Ambiance urbaine / ville (architecture, rue, skyline).",
    "festif_colore": "Ambiance festive et colorée, dynamique.",
    "elegant_sombre": "Ambiance élégante et sombre, sophistiquée, tons profonds.",
    "minimaliste": "Fond minimaliste, très épuré, beaucoup d'espace négatif.",
    "from_images": "Inspire-toi visuellement des images de référence fournies par le client (jointes) pour les couleurs, textures et l'ambiance du fond.",
}


def get_product_config(raw_product_type: str) -> Dict[str, str]:
    p_lower = raw_product_type.lower()
    if "flyer a5" in p_lower:
        return PRODUCT_DIMENSIONS["flyer_a5"]
    elif "flyer a6" in p_lower:
        return PRODUCT_DIMENSIONS["flyer_a6"]
    elif "dépliant" in p_lower or "depliant" in p_lower:
        return PRODUCT_DIMENSIONS["depliant"]
    elif "marque-page" in p_lower or "bookmark" in p_lower:
        return PRODUCT_DIMENSIONS["bookmark"]
    else:
        return PRODUCT_DIMENSIONS["business_card"]


# -----------------------------------------------------------------------
# 2. SCHÉMAS PYDANTIC
# -----------------------------------------------------------------------
class DesignRequest(BaseModel):
    contact_nom: Optional[str] = ""
    contact_prenom: Optional[str] = ""
    product_type: str
    business_type: str
    brand_name: str
    slogan: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    logo_url: Optional[str] = None
    style_preference: Optional[str] = ""
    content_recto: Optional[str] = ""
    content_verso: Optional[str] = ""
    extra_details: Optional[str] = ""
    print_mode: Optional[str] = "recto_verso"
    brand_colors: Optional[List[str]] = None
    background_style: Optional[List[str]] = None
    background_images: Optional[List[str]] = None 

class ColorPalette(BaseModel):
    primary: str
    secondary: str
    background: str
    text: str


class GeminiSingleSideSchema(BaseModel):
    svg_code: str = Field(description="Le code SVG complet et valide")
    color_palette: ColorPalette


class DesignResponseSchema(BaseModel):
    svg_front: str
    svg_back: str
    color_palette: ColorPalette
    print_mode: str


class PdfExportRequest(BaseModel):
    svg_front: str
    svg_back: Optional[str] = ""


# -----------------------------------------------------------------------
# 3. UTILITAIRES - SVG
# -----------------------------------------------------------------------
def sanitize_svg_code(raw_svg: str) -> str:
    clean = re.sub(r"```xml\s*", "", raw_svg)
    clean = re.sub(r"```svg\s*", "", clean)
    clean = re.sub(r"```\s*", "", clean)
    return clean.strip()


def validate_svg(svg_content: str) -> bool:
    try:
        ET.fromstring(svg_content)
        return True
    except ET.ParseError:
        return False


def normalize_print_mode(raw_mode: Optional[str]) -> str:
    mode = (raw_mode or "recto_verso").strip().lower()
    if mode not in VALID_PRINT_MODES:
        mode = "recto_verso"
    return mode


# -----------------------------------------------------------------------
# 3bis. CORRECTION DU CENTRAGE DES TEXTES MULTI-TSPAN
# -----------------------------------------------------------------------
SVG_NS = "[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)"
ET.register_namespace("", SVG_NS)

_FONT_CACHE: Dict[Tuple[int, bool], "ImageFont.FreeTypeFont"] = {}

_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_font(font_size: float, bold: bool):
    key = (max(int(round(font_size)), 1), bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _FONT_BOLD if bold else _FONT_REGULAR
    try:
        font = ImageFont.truetype(path, key[0])
    except OSError:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _is_bold(weight: Optional[str]) -> bool:
    if not weight:
        return False
    weight = str(weight).strip().lower()
    if weight in ("bold", "bolder"):
        return True
    return weight.isdigit() and int(weight) >= 600


def _parse_letter_spacing(el) -> float:
    val = el.get("letter-spacing")
    if not val:
        return 0.0
    val = val.strip().replace("px", "")
    try:
        return float(val)
    except ValueError:
        return 0.0


def _text_width(text: str, font_size: float, bold: bool, letter_spacing: float = 0.0) -> float:
    if not text:
        return 0.0
    font = _load_font(font_size, bold)
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    if letter_spacing:
        width += letter_spacing * max(len(text) - 1, 0)
    return width


def fix_centered_tspans(svg_content: str) -> str:
    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError:
        return svg_content

    def qname(tag: str) -> str:
        return f"{{{SVG_NS}}}{tag}"

    for text_el in root.iter(qname("text")):
        tspans = list(text_el.findall(qname("tspan")))
        if len(tspans) < 2:
            continue

        text_anchor = text_el.get("text-anchor") or tspans[0].get("text-anchor")
        if text_anchor != "middle":
            continue

        base_x_raw = text_el.get("x")
        if base_x_raw is None:
            continue
        try:
            base_x = float(base_x_raw)
        except ValueError:
            continue

        base_font_size = float(text_el.get("font-size", "16") or 16)
        base_weight = text_el.get("font-weight")
        base_spacing = _parse_letter_spacing(text_el)

        lines = []
        current = []
        for i, ts in enumerate(tspans):
            starts_new_line = ts.get("dy") is not None and i != 0
            if starts_new_line and current:
                lines.append(current)
                current = []
            current.append(ts)
        if current:
            lines.append(current)

        for line in lines:
            widths = []
            for ts in line:
                fsize = float(ts.get("font-size", base_font_size) or base_font_size)
                bold = _is_bold(ts.get("font-weight", base_weight))
                spacing = _parse_letter_spacing(ts) or base_spacing
                widths.append(_text_width(ts.text or "", fsize, bold, spacing))

            total_width = sum(widths)
            cursor = base_x - total_width / 2.0

            for ts, w in zip(line, widths):
                ts.set("x", f"{cursor:.2f}")
                ts.attrib.pop("text-anchor", None)
                cursor += w

        text_el.attrib.pop("text-anchor", None)

    return ET.tostring(root, encoding="unicode")


# -----------------------------------------------------------------------
# 4. ENDPOINTS
# -----------------------------------------------------------------------
@app.post("/generate-design", response_model=DesignResponseSchema)
async def generate_design(req: DesignRequest):
    if not client:
        raise HTTPException(
            status_code=500,
            detail="Clé API GEMINI_API_KEY non configurée sur le serveur.",
        )

    has_custom_logo = bool(req.logo_url)
    dim_config = get_product_config(req.product_type)
    print_mode = normalize_print_mode(req.print_mode)

    background_keys = [k for k in (req.background_style or []) if k in BACKGROUND_STYLE_LABELS] or ["automatique"]
    background_instruction = "- Fond / ambiance souhaités : " + " ".join(
        BACKGROUND_STYLE_LABELS[k] for k in background_keys
    )

    reference_image_parts = []
    if "from_images" in background_keys and req.background_images:
        for img_data in req.background_images[:3]:
            try:
                raw = img_data.split(",", 1)[1] if img_data.startswith("data:") else img_data
                image_bytes = base64.b64decode(raw)
                mime_type = "image/png"
                if img_data.startswith("data:") and ";base64" in img_data:
                    mime_type = img_data.split(";")[0].replace("data:", "") or "image/png"
                reference_image_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            except Exception as img_err:
                print(f"[Image de référence ignorée] Décodage impossible : {img_err}")

    brand_colors_clean = [c.strip() for c in (req.brand_colors or []) if c and c.strip()][:3]
    brand_colors_instruction = (
        f"- Couleurs imposées par le client (PRIORITAIRES : utilise-les dans le SVG "
        f"et reflète-les fidèlement dans color_palette) : {', '.join(brand_colors_clean)}"
        if brand_colors_clean
        else "- Aucune couleur imposée par le client : choisis une palette cohérente avec le style visuel demandé."
    )

    contact_full_name = " ".join(
        p.strip() for p in [req.contact_prenom, req.contact_nom] if p and p.strip()
    )
    contact_instruction = (
        f"- Nom du contact à afficher (optionnel, si vide ne rien afficher) : {contact_full_name}. "
        f"Utilise IMPÉRATIVEMENT id=\"svg-contact-name\" sur la balise <text> correspondante. "
        f"Si le prénom + nom sont longs et ne tiennent pas sur une seule ligne avec une marge correcte, "
        f"place le prénom sur une ligne séparée en dessous du nom (utilise des <tspan> avec dy)."
        if contact_full_name else
        "- Aucun nom de contact fourni : ne pas afficher de nom de personne."
    )

    async def generate_single_side(side_label: str) -> Tuple[str, ColorPalette]:
        is_large_format = "xbanner" in req.product_type.lower() or "banner" in req.product_type.lower()
        
        layout_guideline = (
            "- RÈGLE DE MISE EN PAGE GRAND FORMAT (X-Banner / Face unique) : Tout doit être présent sur cette face unique. "
            "Place le logo et le nom de la marque bien en haut, le contenu principal/prestation au centre, "
            "et les coordonnées (Téléphone, E-mail) de manière visible tout en bas."
            if is_large_format else
            f"- RÈGLE DE MISE EN PAGE ({side_label.upper()}) : " + (
                "Sois percutant (Logo, Titre, Slogan, Accroche marketing principale)." if side_label == "recto"
                else "Structure proprement les informations pratiques, les détails libres et les coordonnées de contact."
            )
        )
        
        side_content = req.content_recto if side_label == "recto" else req.content_verso
        content_instruction = (
            f"- Informations à afficher sur cette face (transcris-les fidèlement, n'invente RIEN d'autre) : {side_content}"
            if side_content and side_content.strip()
            else "- Aucune information spécifique fournie pour cette face : reste sobre, n'invente pas de fausses informations."
        )

        prompt = f"""
Tu es un expert en design graphique et impression professionnelle (Web2Print).
Génère le design SVG pour la face **{side_label.upper()}** d'un support : {req.product_type}.

IMPOSEE - FORMAT ET ORIENTATION SVG :
- Doit IMPÉRATIVEMENT utiliser le viewBox exact suivant : `viewBox="{dim_config['viewBox']}"`
- Format d'orientation : {dim_config['aspect']}
- {layout_guideline}

Informations de la marque :
- Nom / Enseigne : {req.brand_name}
- Secteur / Activité à afficher explicitement : {req.business_type}
- Slogan : {req.slogan}
- {contact_instruction}
- {content_instruction}
- Coordonnées de contact :
      * Téléphone : Libellé "Tél :" et valeur "{req.phone}"
      * Email : Libellé "Email :" et valeur "{req.email}"
- Style visuel souhaité : {req.style_preference} {brand_colors_instruction} {background_instruction}
- Consignes particulières : {req.extra_details}
- Prescriptions Logo : "Crée un bloc de logo vectoriel strictement vertical et espacé : 1) Place le cercle avec son icône en haut. 2) Place le nom de la marque ('{req.brand_name}') en dessous avec un espace vertical obligatoire (text-anchor='middle'). 3) Place le sous-titre encore plus bas avec un espacement net. Interdiction formelle de superposer les textes ou de les aligner sur la même ligne horizontale."

RÈGLES DE RENDU ET OPTIMISATION SVG :
1. **ULTRA-CONCIS** : Pour les flyers, utilise uniquement 3 à 5 éléments graphiques majeurs maximum (`<rect>`, `<circle>`, `<text>`). N'utilise **JAMAIS** de balises `<path>` complexes avec des centaines de coordonnées.
2. Utilise l'attribut `viewBox="{dim_config['viewBox']}"` sur la balise racine `<svg>`. Ne fixe PAS de width/height absolus.
3. Le code SVG doit être un XML 1.0 valide. FERME IMPÉRATIVEMENT TOUTES LES BALISES.
4. Ne JAMAIS insérer le caractère '&' non échappé dans les textes (remplace-le par '&amp;').

RÈGLES DE CONCEPTION GRAPHIQUE STRICTES :
1. **HIÉRARCHIE TEXTUELLE** : Affiche clairement dans l'ordre vertical : 1) Le logo et le nom de la marque, 2) Le secteur d'activité, 3) Le slogan (si présent), 4) Le contenu spécifique et les contacts en bas.
2. **ZÉRO LIGNE PARASITE** : Interdiction absolue de tracer des lignes aléatoires ou obliques non sollicitées.
"""

        last_exception = None

        for model_name in FALLBACK_MODELS:
            print(f"[Tentative] Génération du {side_label} avec le modèle : {model_name}")

            for attempt in range(2):
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model_name,
                        contents=[prompt, *reference_image_parts] if reference_image_parts else prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GeminiSingleSideSchema,
                            temperature=0.5,
                            max_output_tokens=8192,
                        ),
                    )

                    if response.text:
                        parsed = GeminiSingleSideSchema.model_validate_json(response.text)
                        clean_svg = sanitize_svg_code(parsed.svg_code)

                        if has_custom_logo and req.logo_url:
                            clean_svg = clean_svg.replace("PLACEHOLDER_LOGO", req.logo_url)

                        if validate_svg(clean_svg):
                            return clean_svg, parsed.color_palette
                        else:
                            print(f"[XML Invalide] Tentative {attempt + 1} sur {model_name}. Relance...")

                except ValidationError as ve:
                    print(f"[JSON Tronqué/Invalide] Tentative {attempt + 1} sur {model_name}. Relance...")
                    last_exception = ve
                except APIError as e:
                    last_exception = e
                    err_str = str(e)
                    if (
                        getattr(e, "code", None) in [429, 503]
                        or "RESOURCE_EXHAUSTED" in err_str
                        or "UNAVAILABLE" in err_str
                        or "high demand" in err_str.lower()
                    ):
                        wait_time = (attempt + 1) * 2
                        print(f"[Surcharge / Quota] Modèle {model_name}. Pause de {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        break
                except Exception as e:
                    last_exception = e
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower():
                        wait_time = (attempt + 1) * 2
                        print(f"[Surcharge 503] Pause de {wait_time}s avant de relancer...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"[Erreur] Tentative {attempt + 1} sur {model_name}: {err_str}")
                        break

        raise HTTPException(
            status_code=500,
            detail=f"Échec de génération d'un SVG valide pour le {side_label}. Message : {str(last_exception)}",
        )

    try:
        svg_front, palette_front = await generate_single_side("recto")

        svg_back = ""
        palette_back = palette_front

        if print_mode == "recto_verso":
            await asyncio.sleep(1.5)
            svg_back, palette_back = await generate_single_side("verso")
        else:
            print("[Mode Recto seul] Génération du verso ignorée.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur d'exécution globale : {str(e)}"
        )

    return DesignResponseSchema(
        svg_front=svg_front,
        svg_back=svg_back,
        color_palette=palette_front,
        print_mode=print_mode,
    )


@app.post("/export-pdf")
async def export_pdf(payload: PdfExportRequest):
    try:
        if not payload.svg_front or not payload.svg_front.strip():
            raise HTTPException(
                status_code=400,
                detail="Le SVG recto est manquant : impossible de générer le PDF.",
            )

        svg_front_fixed = fix_centered_tspans(payload.svg_front)

        pdf_front_bytes = cairosvg.svg2pdf(
            bytestring=svg_front_fixed.encode("utf-8")
        )

        merger = PdfWriter()
        merger.append(io.BytesIO(pdf_front_bytes))

        has_back_content = bool(payload.svg_back and payload.svg_back.strip() != "")

        if has_back_content:
            svg_back_fixed = fix_centered_tspans(payload.svg_back)
            pdf_back_bytes = cairosvg.svg2pdf(
                bytestring=svg_back_fixed.encode("utf-8")
            )
            merger.append(io.BytesIO(pdf_back_bytes))

        output_stream = io.BytesIO()
        merger.write(output_stream)
        merger.close()

        filename = "impression_recto_verso.pdf" if has_back_content else "impression_recto.pdf"

        return Response(
            content=output_stream.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération du PDF : {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="127.0.0.1", port=8000, reload=True
    )