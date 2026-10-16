# ARJUN — pitch notes

Questions judges actually ask, and answers that hold up. Each one has a short
answer to say out loud and the detail to fall back on if they push.

**The single most useful habit:** when a question has a number in it, give the
number. "How fast?" → "2.4 seconds from the floor to the guard's phone." Vague
answers sound like a demo; specific ones sound like a system.

---

## Privacy

### "If it can't see faces, how is this different from just blurring the video?"

**Say:** "Blurring is a filter applied to a recording you still have. We never
have the recording. The dashboard isn't a processed camera image — it's drawn
from geometry onto a blank canvas. The renderer is never handed the frame."

**Detail:** `PrivacyRenderer.render()` takes a `FrameWorld` — boxes, track ids,
joint coordinates — and `cv2`-draws onto a fresh black canvas. There is no code
path where the input frame becomes a layer. A blur can be removed, reversed, or
forgotten about in a config file. There's nothing here to un-blur.

### "What stops someone just turning the privacy mode off?"

**Say:** "There isn't a privacy mode to turn off. There's a raw view, it needs
the staff PIN, it lasts ten minutes, and using it writes a line in the audit
log that appears on the dashboard in front of everyone. Staff can see faces
when they need to. They can't do it quietly."

**Detail:** raw frames aren't even JPEG-encoded until a valid token asks for
them (`CameraWorker.request_raw`). Every unlock, every raw stream open and
every clip fetch is an `audit` row and a live WebSocket push.

### "You still store clips. So you do store footage."

**Say:** "Yes — fifteen seconds per incident, and that's the honest part of the
design. If someone falls down a stairwell, a guard needs to see what happened.
What we don't store is the other 99.99% of the day. Today that's *(read the
ledger)* clips. Compare that to a normal system, which would have about eight
thousand hours a week for a campus this size."

**Detail:** 10 s before the trigger plus 5 s after, written by the only
component in the codebase that writes pixels, deleted after 7 days, capped at
60 files.

### "Could I reconstruct a person's day from the event database?"

**Say:** "No, and that's a design decision rather than an accident. There's no
appearance descriptor anywhere in the system, so if you walk past two cameras
you're two unrelated track numbers. We deliberately didn't build
re-identification, because that's the thing that would turn this back into
surveillance."

**Detail:** track ids come from ByteTrack and are per-camera and per-session.
`TrackStore` deletes a track six seconds after it stops being seen. Nothing
persists across a restart except events, zones and the audit log.

### "The skeleton has a gait. Gait is biometric."

**Say:** "That's a fair challenge, and it's the strongest privacy objection
anyone's made to us. Gait recognition is real — but it needs a gallery of
labelled walks to match against, and a persistent store of walk sequences. We
have neither: histories are eight seconds and then gone, and nothing is
written to disk. Could someone fork this and add gait matching? Yes. They'd be
building a different system, and they'd have to add storage we deliberately
didn't."

**Detail:** this is the right answer to give — judges respect "here is the
limit of our claim" far more than a flat denial.

---

## Accuracy and false positives

### "How accurate is the fall detection?"

**Say:** "We haven't got a labelled dataset of real falls, so we won't quote a
percentage we can't defend — and you should distrust anyone at a hackathon who
does. What we can show is what it rejects. Sitting down drops the hips fast,
but the torso stays vertical. Bending over is a wide box with a vertical torso.
A stumble and recovery doesn't last two seconds. Each of those is a test in
the suite."

**Detail:** `tests/test_fall.py` has eleven cases, including sitting, walking,
stumble-and-recover, somebody already on the floor, and the same fall at a
third the scale.

### "What's your false positive rate?"

**Say:** "We don't have a number from real deployment, so here's the design
principle instead: every rule has a confirmation window, and the windows are
the reason the alerts are worth reading. A fall needs two seconds on the floor.
A crowd needs to hold for 2.5. Panic needs half a second. Those cost response
time, and they buy an alert a guard still trusts in week three — a rule nobody
trusts is a rule nobody acts on."

### "Isn't two seconds a long time if someone's hurt?"

**Say:** "It is, and it's a deliberate trade. Without it, every person who sits
on the floor is an ambulance call, and within a day staff stop looking at the
alerts — at which point the response time is however long until someone walks
past, which is what we were trying to fix. Two seconds plus a second of
network still beats 'until somebody notices' by orders of magnitude."

### "What about a camera pointed at a crowded lecture hall?"

**Say:** "Crowding hurts us in two ways: occlusion breaks tracking, and the
rules that depend on duration restart when a track number changes. That's a
real limitation, and it's why crowd limits are per-zone rather than global —
you tune the zone to the camera."

---

## Technical depth

### "Why YOLO-pose every frame, but the object detector only every third?"

**Say:** "Because people are the safety-critical subject and a fall lasts half
a second — you can't sample that at four hertz. A bag that's been motionless
for thirty seconds doesn't need finding sixty times a second. That one choice
roughly doubles the frame rate."

**Detail:** the brief suggested detector + MediaPipe Pose per person crop.
MediaPipe Pose handles one subject per call, so six people costs seven
inferences per frame. Measured 46 ms for the inverted arrangement versus
single-digit fps for the obvious one.

### "Why not MediaPipe for pose then?"

**Say:** "We use MediaPipe where it's genuinely the best tool — hand landmarks
for the Signal for Help, and the face mesh's 3D transform matrix for head yaw.
Both run on small crops, only when the feature that needs them is switched on,
behind a cheap gate. The SOS rule checks 'is this person's wrist above their
shoulder' from the pose before it ever loads a hand model. In a room of thirty
seated students the hand model runs zero times a second."

### "How do your thresholds work at different distances?"

**Say:** "Nothing is in pixels. Speeds are in body heights per second,
distances in multiples of the subject's own height. A person two metres from
the lens is four times taller in pixels than one at the end of a corridor — if
you threshold in pixels you've built a system that only works at one distance.
There's a test that runs the same fall at a third the scale and expects the
same verdict."

### "What happens when a camera disconnects?"

**Say:** "The tile says Reconnecting and that worker retries with a widening
backoff. One thread per camera, nothing shared — ByteTrack's state actually
lives inside the model object, so two cameras sharing a model would swap
identities. A camera failing can't affect another one, and a rule that throws
gets muted for a few seconds while the other ten keep running."

### "Walk me through a bug you found."

*This is the best question you can get. Have one ready.*

**Say:** "Line counting. There's a dead-band around the line so someone
standing on the threshold isn't counted sixty times a second. But we were
storing in-band readings as the 'previous side', so if a stride put a foot
*exactly* on the line, that became the reference and the crossing on the far
side was silently skipped. The first test passed by luck of stride length; the
second one caught it. The band is hysteresis now — only confident sides are
remembered."

**Or:** "The unattended-bag rule. We first wrote 'nobody within two metres' —
and in a busy corridor somebody is within two metres almost continuously, so
the reclaim timer reset forever and the rule would have **never fired at all**.
That's worse than not having it, because you'd believe you had it. Custody now
means near it *and* stopped. There's a test that sweeps a pedestrian past the
bag every few seconds and insists the alert still arrives."

### "Did you test this properly?"

**Say:** "91 tests, they run in about a second, and they need no camera and no
model weights. Each rule is driven by synthetic skeletons on a virtual clock
against its real thresholds — a thirty-second loitering timer is tested in two
milliseconds. That's the thing that let us actually iterate on the rules in a
day."

---

## Scalability and cost

### "Would this work on a campus with two hundred cameras?"

**Say:** "Not on one laptop — we get about three cameras per machine on CPU.
The architecture is per-camera threads with no shared state, so it scales by
adding boxes, not by rewriting. Realistically: one small machine per eight to
twelve cameras with a GPU, or a Jetson at the camera itself, which is where we
actually want to go — if inference happens in the camera, frames never travel
at all."

### "What does it cost to run?"

**Say:** "The vision stack is free and offline — Ultralytics and MediaPipe
weights, about 30 MB, downloaded once. Telegram is free. The only metered thing
is Claude, and it's only called when someone asks a question or presses
'write the report' — it isn't in the frame loop. A day of heavy demo use is
cents. And the whole thing runs with no keys at all; you lose written answers,
not safety."

### "What hardware does a pilot need?"

**Say:** "Any machine that can run this laptop's workload — and the cameras you
already own, if they do RTSP. We tested with a laptop webcam and a phone over
DroidCam, because that's what a student team has."

---

## Ethics of exam mode

### "Isn't exam mode exactly the surveillance you're arguing against?"

**Say:** "It's the one feature we argued about internally, so here's where we
landed. It's off by default, opt-in per camera, and it cannot name anybody — it
says 'Desk 3: phone visible for four seconds, for invigilator review'. It can't
tell you who's at desk three, because the system doesn't know and has no way to
find out. It's a prompt for a human to walk over and look, which is what
invigilators already do. We write flags, not findings."

**If they push:** "And if your institution doesn't want it, don't draw desk
zones. The feature does nothing without them."

### "What if a school used this for attendance or discipline?"

**Say:** "They couldn't, and that's the point of refusing identity. Attendance
needs names. Discipline needs to know who. We don't have either, and adding
them would mean building the re-identification we deliberately left out. The
README has a section listing what this should not be used for, and we'd rather
be judged on having written it than on pretending the question doesn't exist."

### "Head-turn detection in an exam feels like a stretch."

**Say:** "Agreed, and we log it at *info* severity for exactly that reason.
Students look at clocks. It's there so an invigilator reviewing a session has
the full picture, not so anyone gets accused. If we were deploying this
tomorrow it's the first feature we'd offer to turn off."

---

## The hard ones

### "Why skeletons? If a student assaults someone, can you still catch them?"

**Say:** "Yes, and this is the misconception worth clearing up. Skeletons are
how we *notice*. When something happens, the system saves a fifteen-second
clip with full raw video — ten seconds before, five after. A guard with the PIN
opens it and sees exactly who it was.

The difference is which direction the default runs. A normal system records
everybody and keeps it in case somebody turns out to matter. This one records
nobody and keeps fifteen seconds in volatile memory in case somebody turns out
to need help. **The evidence exists when there's an incident. It doesn't exist
the rest of the time.**"

### "Isn't this just a worse version of what's commercially available?"

**Say:** "Commercially available systems are better at detection than we are —
they have datasets and years. What they don't have is this architecture. The
industry answer to privacy is a retention policy and an access control list,
both of which are promises. Ours is that the data isn't there. That's not a
feature we added; it's the shape of the pipeline, and you can verify it by
reading one function."

### "What would you do with another month?"

**Say:** "Three things, in order. One: a labelled dataset of real falls, so
those thresholds are measured instead of tuned by hand. Two: on-device
inference on a Jetson so frames never leave the camera — that closes the last
gap in the privacy argument, because right now frames do travel over the
network to the machine running this. Three: a proper privacy-impact assessment
and a student-facing page explaining exactly what the cameras do, because a
system like this should be explainable to the people it watches over."

### "What are you most proud of?"

**Say:** "That the privacy claim is structural rather than promised. We could
have built a normal detection system and put a blur on the output, and from
three metres away the demo would look identical. Instead the renderer is never
given the frame, the rules can't reach the pixels, and the only component that
can write video is the one that runs when somebody needs help. You can check
all three by reading the code, and that's a different kind of claim from
'we take privacy seriously'."

---

## Phrases worth having ready

* "Normal CCTV records thousands of innocent students to catch one incident.
  This watches no one, until someone gives it a reason to."
* "CCTV is a witness. This is a first responder."
* "Zero hours stored — not as a policy, but because there's no code in it that
  can."
* "Safety doesn't require watching people. It requires noticing events. Those
  aren't the same problem."
* "We write flags, not findings."
* "The evidence exists when there's an incident. It doesn't exist the rest of
  the time."
* "A rule nobody trusts is a rule nobody acts on."

## Things not to say

* Any accuracy percentage you can't source. One "about 95%" and a judge who
  asks "measured how?" costs you the technical-depth mark.
* "It's fully production ready." Say what a pilot would need instead — it reads
  as more competent, not less.
* "It's 100% private." Say what's held, for how long, and who can unlock it.
  The specific answer is stronger than the absolute one.
* Don't oversell the simulated scenes. Lead with "scripted scene, running
  through the identical detectors" — volunteering it turns a weakness into a
  demonstration of how the rules are tested.
