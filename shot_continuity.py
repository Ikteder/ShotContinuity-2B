"""Generate original Blender shot-continuity fixtures without external data."""
import json
import random

SYSTEM = (
    "You check continuity across two adjacent shots. Compare the tracked values in shot A and shot B. "
    "The director's note authorizes only changes that actually happen between the shots. "
    "A rejected idea, rehearsal, camera change, or plan for a later shot does not authorize a change. "
    "Return only JSON as {\"issues\":[{\"field\":\"field name\",\"restore\":\"shot A value\"}]}. "
    "Include only changed fields that should be restored. Order issues by field. Use an empty list if clean."
)

TRAIN_SETS = (
    ("Subway farewell", "Mira", ("jacket", "key", "scarf", "cup"), ("denim", "right hand", "loose", "half full"),
     ("tan", "coat pocket", "knotted", "empty")),
    ("Snow lantern", "Tavi", ("boots", "lantern", "glove", "door"), ("muddy", "lit", "left hand", "closed"),
     ("clean", "unlit", "right hand", "open")),
    ("Kitchen rehearsal", "Sam", ("apron", "notebook", "plate", "window"), ("gray", "counter", "blue", "ajar"),
     ("orange", "floor", "white", "shut")),
    ("Robot station", "Pip", ("badge", "antenna", "battery", "toolbox"), ("silver", "upright", "full", "latched"),
     ("gold", "folded", "low", "open")),
    ("Cafe rain", "Jo", ("umbrella", "bottle", "bag", "wristwatch"), ("red", "table", "zipped", "left wrist"),
     ("yellow", "shelf", "open", "right wrist")),
    ("Workshop return", "Ivo", ("helmet", "brush", "canvas", "stool"), ("white", "jar", "dry", "by door"),
     ("black", "floor", "wet", "under window")),
    ("Alley chase", "Luz", ("shoe", "flashlight", "map", "backpack"), ("left foot", "on", "folded", "shoulder"),
     ("off", "off", "unfolded", "floor")),
    ("Library blackout", "Ari", ("glasses", "book", "candle", "curtain"), ("worn", "open", "unlit", "closed"),
     ("on desk", "closed", "lit", "open")),
    ("Harbor crossing", "Ren", ("life vest", "oar", "rope", "radio"), ("fastened", "port side", "coiled", "off"),
     ("loose", "starboard", "uncoiled", "on")),
    ("Desert outpost", "Sol", ("canteen", "sleeve", "binoculars", "crate"), ("sealed", "rolled up", "neck", "shut"),
     ("open", "down", "ledge", "open")),
)
TEST_SETS = (
    ("Moon greenhouse", "Niko", ("visor", "seed bag", "soil tray", "water valve"), ("fogged", "sealed", "dry", "off"),
     ("clear", "open", "wet", "on")),
    ("Museum heist", "Rin", ("mask", "velvet case", "medallion", "display panel"),
     ("on face", "closed", "pedestal", "green"), ("on neck", "open", "glove", "red")),
    ("River puppet show", "Amal", ("puppet", "peg", "paint", "stage cloth"), ("up", "left", "dry", "folded"),
     ("down", "right", "wet", "spread")),
    ("Pixel train", "Emi", ("ticket", "luggage", "cap", "carriage light"),
     ("pocket", "overhead", "forward", "on"), ("hand", "seat", "backward", "off")),
    ("Festival tower", "Thea", ("banner", "medal", "camera strap", "shutter"),
     ("rolled", "neck", "left side", "locked"), ("unrolled", "table", "right side", "unlocked")),
    ("Paper dragon cave", "Oona", ("tail", "torch", "satchel", "hinge"),
     ("curled", "burning", "closed", "oiled"), ("straight", "smoking", "open", "dry")),
)
FINAL_SETS = (
    ("Lighthouse study", "Jules", ("coat", "mug", "desk lamp", "shade"),
     ("navy", "desk", "off", "raised"), ("gray", "in hand", "on", "lowered")),
    ("Aquarium opening", "Cleo", ("ladder", "flippers", "tank cover", "towel"),
     ("wall", "dry", "flat", "hook"), ("tank", "wet", "tilted", "bench")),
    ("Mountain observatory", "Kavi", ("dome", "chair", "journal", "red light"),
     ("closed", "inside", "shut", "dim"), ("open", "outside", "open", "bright")),
    ("Chalk theater", "Sana", ("chalk", "chalkboard", "prop sword", "sash"),
     ("blue", "blank", "stage left", "belt"), ("white", "drawn on", "stage right", "shoulder")),
)

TRAIN_CHANGE = (
    "Between the shots, {hero} changes the {field} to {value}.",
    "The script calls for one on-screen change before B. The {field} is now {value}.",
    "We clearly see the {field} become {value} on the cut.",
    "Keep the planned edit to the {field}, which ends up {value} after the cut.",
    "The action between setups puts the {field} at {value}.",
    "By the time the next camera rolls, {hero} has deliberately made the {field} {value}.",
    "B begins immediately after {hero} sets the {field} to {value}.",
    "The {field} change is part of the scene. Leave it {value} in the second shot.",
)
TEST_CHANGE = (
    "The insert we actually kept shows the {field} ending at {value}.",
    "Motivated change. B opens just after {hero} makes the {field} {value}.",
    "A time cut explains the {field}; its new state is intentionally {value}.",
)
TRAIN_FAKE = (
    "The {field} was supposed to be {value} in a later shot, but not in B.",
    "A rehearsal tried the {field} at {value}, but that take was discarded.",
    "Use a new camera angle. Don't make the {field} {value}.",
)
TEST_FAKE = (
    "Scratch that idea about changing the {field} to {value}.",
    "The storyboards mention {value} for the {field}. That beat was moved after B.",
    "The camera angle changed. The proposed change to the {field} ({value}) was vetoed.",
)
FINAL_CHANGE = (
    "B follows an action beat in which {hero} deliberately leaves the {field} {value}.",
    "Continuity has signed off on the {field} being {value} from B onward.",
    "Before we cut, the {field} really does end up {value}.",
)
FINAL_FAKE = (
    "The note about the {field} being {value} referred to an alternate edit we aren't using.",
    "The planned transition to {value} for the {field} should happen only after this pair.",
)


def generate(split):
    training = split == "train"
    final = split == "final"
    rng = random.Random(2701 if training else 3961 if final else 9803)
    settings = TRAIN_SETS if training else FINAL_SETS if final else TEST_SETS
    allow_phrases = TRAIN_CHANGE if training else FINAL_CHANGE if final else TEST_CHANGE
    deny_phrases = TRAIN_FAKE if training else FINAL_FAKE if final else TEST_FAKE
    modes = ("clean", "authorized", "unexplained", "mixed", "rejected", "several")
    for scene, hero, fields, original, alternate in settings:
        for index in range(24 if training else 6 if final else 9):
            mode = modes[index % len(modes)]
            shuffled = list(range(len(fields)))
            rng.shuffle(shuffled)
            authorized = set(shuffled[:2] if mode == "authorized" else shuffled[:1] if mode in ("mixed", "several") else [])
            bad = set(shuffled[1:3] if mode == "several" else shuffled[1:2] if mode == "mixed"
                      else shuffled[:1] if mode in ("unexplained", "rejected") else [])
            old = dict(zip(fields, original))
            options = dict(zip(fields, alternate))
            # Swap which state was established in A to prevent memorizing which value to restore.
            for field in fields:
                if rng.randrange(2):
                    old[field], options[field] = options[field], old[field]
            next_shot = {fields[i]: options[fields[i]] if i in authorized | bad else old[fields[i]]
                         for i in shuffled}
            notes = ["Camera reframes a little closer in B."]
            for i in sorted(authorized):
                notes.append(rng.choice(allow_phrases).format(hero=hero, field=fields[i], value=options[fields[i]]))
            if bad and mode in ("mixed", "rejected", "several"):
                suspect = fields[min(bad)]
                notes.append(rng.choice(deny_phrases).format(field=suspect, value=options[suspect]))
            rng.shuffle(notes)
            request = {
                "scene": scene, "actor": hero, "shot_a": old, "shot_b": next_shot,
                "director_note": " ".join(notes),
            }
            expected = {"issues": [{"field": f, "restore": old[f]} for f in sorted(fields[i] for i in bad)]}
            yield {"scene": scene, "mode": mode, "input": request, "expected": expected}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
