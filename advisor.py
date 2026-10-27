"""Rule-based treatment advisor: maps a predicted PlantVillage class to advice.

The deep-learning model only says WHICH disease it sees. This module turns that
label into practical guidance (symptoms, action, prevention) for the farmer.
Advice is general guidance, not a prescription: always follow product labels and
local agriculture-department recommendations.
"""
import re

# Each entry: (keywords matched against the cleaned disease name, advice).
# More specific keywords come first.
_ADVICE = [
    (("healthy",), {
        "kind": "Healthy",
        "symptoms": "No visible signs of disease on this leaf.",
        "treatment": "No treatment needed. Keep up regular watering, balanced fertiliser and weekly scouting.",
        "prevention": "Remove weeds, avoid watering late in the day, and check leaves regularly.",
    }),
    (("northern leaf blight",), {
        "kind": "Fungal",
        "symptoms": "Long, cigar-shaped grey-green to tan lesions on the leaves.",
        "treatment": "Apply a recommended foliar fungicide at first signs, before the disease reaches upper leaves.",
        "prevention": "Use resistant hybrids, rotate crops and bury or remove infected crop residue.",
    }),
    (("cercospora", "gray leaf spot"), {
        "kind": "Fungal",
        "symptoms": "Rectangular tan to grey lesions that run parallel to the leaf veins.",
        "treatment": "Use a labelled foliar fungicide if lesions are spreading up the plant before tasseling.",
        "prevention": "Rotate crops, manage residue and choose resistant hybrids.",
    }),
    (("common rust",), {
        "kind": "Fungal",
        "symptoms": "Small round brick-red to brown pustules on both sides of the leaf.",
        "treatment": "Often minor; use a labelled fungicide only if infection is early and heavy on a susceptible hybrid.",
        "prevention": "Plant resistant hybrids and sow early.",
    }),
    (("cedar apple rust",), {
        "kind": "Fungal",
        "symptoms": "Bright yellow-orange spots on the upper leaf surface with tube-like growths underneath.",
        "treatment": "Apply a labelled fungicide from the pink-bud stage; remove nearby juniper or cedar hosts if possible.",
        "prevention": "Plant resistant varieties and keep trees away from junipers.",
    }),
    (("scab",), {
        "kind": "Fungal",
        "symptoms": "Olive-green to brown scabby spots on leaves and fruit; leaves may curl and drop early.",
        "treatment": "Remove fallen leaves and spray a labelled protectant fungicide (for example sulphur-based) from green-tip stage.",
        "prevention": "Prune for airflow, rake leaves in autumn and choose resistant varieties.",
    }),
    (("esca", "black measles"), {
        "kind": "Fungal (wood disease)",
        "symptoms": "Tiger-stripe pattern between leaf veins and dark spots on the berries.",
        "treatment": "No cure. Prune infected wood in dry weather, disinfect tools and remove severely affected vines.",
        "prevention": "Protect pruning wounds and avoid stressing the vines.",
    }),
    (("isariopsis",), {
        "kind": "Fungal",
        "symptoms": "Dark irregular leaf spots with a yellow halo, followed by early leaf fall.",
        "treatment": "Remove infected leaves and spray a copper-based fungicide or Bordeaux mixture as per label.",
        "prevention": "Open up the canopy and avoid overhead irrigation.",
    }),
    (("black rot",), {
        "kind": "Fungal",
        "symptoms": "Dark brown circular leaf spots and black, shrivelled (mummified) fruit.",
        "treatment": "Prune out infected shoots, remove mummified fruit and apply a labelled protectant fungicide.",
        "prevention": "Practise good sanitation, keep the canopy airy and avoid wetting foliage.",
    }),
    (("haunglongbing", "citrus greening"), {
        "kind": "Bacterial (spread by psyllid insects)",
        "symptoms": "Blotchy yellow mottling of leaves and small, lopsided, bitter fruit.",
        "treatment": "No cure. Remove and destroy infected trees, control the psyllid vector and inform the agriculture department.",
        "prevention": "Plant certified disease-free saplings and control psyllids.",
    }),
    (("powdery mildew",), {
        "kind": "Fungal",
        "symptoms": "White powdery coating on leaves, with curling and stunted growth.",
        "treatment": "Remove badly affected leaves and spray sulphur, neem oil or a labelled fungicide.",
        "prevention": "Give plants space and sunlight, avoid excess nitrogen and use resistant varieties.",
    }),
    (("bacterial spot",), {
        "kind": "Bacterial",
        "symptoms": "Small dark, water-soaked spots that turn brown, often with a yellow halo; spots may appear on fruit.",
        "treatment": "Spray a copper-based bactericide as per label, remove infected debris and do not handle wet plants.",
        "prevention": "Use disease-free seed or seedlings, drip irrigation and 2-3 year crop rotation.",
    }),
    (("early blight",), {
        "kind": "Fungal",
        "symptoms": "Brown spots with concentric rings (bullseye pattern) on older leaves, followed by yellowing.",
        "treatment": "Remove infected lower leaves and apply a labelled fungicide such as mancozeb or chlorothalonil.",
        "prevention": "Rotate crops, mulch, keep plants well fed and spaced.",
    }),
    (("late blight",), {
        "kind": "Oomycete (fast-spreading, urgent)",
        "symptoms": "Water-soaked dark greasy patches, white fuzzy growth underneath in humid weather and rapid collapse.",
        "treatment": "Act immediately: remove and destroy infected plants (do not compost) and use a locally recommended protectant plus systemic fungicide.",
        "prevention": "Use certified seed, keep foliage dry, choose resistant varieties and watch the weather.",
    }),
    (("leaf scorch",), {
        "kind": "Fungal",
        "symptoms": "Small purple spots that enlarge, giving leaf edges a scorched look.",
        "treatment": "Remove infected leaves and apply a labelled fungicide.",
        "prevention": "Renovate old beds, space plants and avoid overhead watering.",
    }),
    (("leaf mold",), {
        "kind": "Fungal",
        "symptoms": "Pale yellow spots on top of the leaf with olive-green to grey velvety mould underneath.",
        "treatment": "Improve ventilation, lower humidity, remove infected leaves and apply a labelled fungicide.",
        "prevention": "Use resistant varieties, space plants and avoid wetting leaves.",
    }),
    (("septoria",), {
        "kind": "Fungal",
        "symptoms": "Many small circular spots with dark borders and grey centres, starting on lower leaves.",
        "treatment": "Remove infected lower leaves and spray copper, mancozeb or chlorothalonil as per label.",
        "prevention": "Rotate crops, mulch the soil and use drip irrigation.",
    }),
    (("spider mite",), {
        "kind": "Pest (mite)",
        "symptoms": "Tiny yellow stippling, fine webbing under the leaves and bronzing in hot, dry weather.",
        "treatment": "Wash leaves with a strong water jet, use neem oil or insecticidal soap, and a labelled miticide for heavy infestations.",
        "prevention": "Avoid dusty, drought-stressed plants and encourage natural predators.",
    }),
    (("target spot",), {
        "kind": "Fungal",
        "symptoms": "Brown spots with concentric rings and yellow halos on leaves and fruit.",
        "treatment": "Remove infected leaves and apply a labelled fungicide.",
        "prevention": "Improve airflow, avoid long periods of leaf wetness and rotate crops.",
    }),
    (("yellow leaf curl", "yellowleaf"), {
        "kind": "Viral (spread by whiteflies)",
        "symptoms": "Upward leaf curling, yellow leaf margins, stunted plants and very few fruits.",
        "treatment": "No cure. Remove infected plants and control whiteflies with yellow sticky traps, neem or a labelled insecticide.",
        "prevention": "Use resistant varieties, insect nets in nurseries and keep weeds down.",
    }),
    (("mosaic",), {
        "kind": "Viral",
        "symptoms": "Light and dark green mottling, distorted leaves and stunted growth.",
        "treatment": "No cure. Remove infected plants and disinfect tools and hands.",
        "prevention": "Use certified seed, avoid handling tobacco before touching plants and choose resistant varieties.",
    }),
]

_GENERIC = {
    "kind": "Unknown",
    "symptoms": "Pattern not covered by the built-in advice table.",
    "treatment": "Isolate the plant and show a sample to your local agriculture officer or Krishi Vigyan Kendra (KVK).",
    "prevention": "Keep foliage dry, remove infected leaves and maintain good field hygiene.",
}


def _clean(text):
    """'Pepper,_bell' -> 'Pepper, bell'; 'Leaf_Mold' -> 'Leaf Mold'."""
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def split_label(class_name):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight').

    Also handles other PlantVillage naming styles, such as
    'Pepper__bell___Bacterial_spot' and 'Tomato_Bacterial_spot'.
    """
    if "___" in class_name:
        crop, _, disease = class_name.partition("___")
    else:
        crop, _, disease = class_name.partition("_")
    return _clean(crop), _clean(disease) or "Unknown"


def pretty_label(class_name):
    crop, disease = split_label(class_name)
    return f"{crop} - {disease}"


def get_advice(class_name):
    """Return a dict with crop, disease, kind, symptoms, treatment, prevention."""
    crop, disease = split_label(class_name)
    key = re.sub(r"[^a-z0-9]+", " ", disease.lower()).strip()
    advice = _GENERIC
    for keywords, entry in _ADVICE:
        if any(k in key for k in keywords):
            advice = entry
            break
    return {"crop": crop, "disease": disease, **advice}
