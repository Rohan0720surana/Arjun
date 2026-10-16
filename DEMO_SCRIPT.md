# ARJUN — three-minute demo script

> **Before you start**
>
> 1. `python run.py --demo` (add `--sim` if you have no camera, or if the venue
>    wifi is hostile — the scripted scenes stage a fall, a bag and a crowd for
>    you on a 78-second loop).
> 2. Dashboard open, full screen, **Demo timings** already on.
> 3. Phone out with Telegram visible, if you configured it. The alert arriving
>    on a phone mid-sentence is the best thirty seconds of this demo.
> 4. Have a backpack and one teammate ready.
>
> Timings below assume two presenters: **A** talks, **B** does things in front
> of the camera. One person can do it; just start the fall earlier.

---

## 0:00 — The hook (20 seconds)

> **A:** "This campus has cameras in every corridor. Right now they're
> recording all of us — every person in this room who walked past one today is
> on a disk somewhere, so that if something happens, someone can go back and
> look.
>
> That's the deal we've accepted: **record thousands of innocent people, to
> catch one incident.**
>
> We think that's a bad deal, and we think it's unnecessary. So we built the
> opposite."

*Gesture at the screen.*

> "This is ARJUN. It watches no one — until someone gives it a reason to."

---

## 0:20 — What they're looking at (25 seconds)

*Point at a camera tile with people moving in it.*

> **A:** "That's a live camera. That's a person walking — but there's no
> person on that screen. There are joints, and a number.
>
> This isn't a blur filter over a video. The camera image **never reaches this
> screen**. We take the frame, work out where the joints are, throw the pixels
> away, and draw the skeleton onto a blank canvas. There is no setting that
> turns the faces back on, because the faces were never sent."

*Point at the camera name, then the gold number above a skeleton.*

> "Person forty-one. That number was invented four seconds ago and it'll be
> gone four seconds after they leave. It means nothing tomorrow, and nothing on
> any other camera."

---

## 0:45 — The fall (35 seconds) — *the heart of the demo*

> **A:** "So what's the point of a camera that refuses to look at people?"

*B walks into frame and lies down on the floor.*

> **A:** *(while B is on the floor, counting)* "Normal CCTV has just recorded
> this. Nobody is watching it. Someone will find him when they walk past."

*Wait. The tile flashes red, the chime sounds, the alert appears.*

> **A:** "Two seconds. Hip dropped sixteen percent of his body height in
> half a second, torso went horizontal, stayed down past two seconds — that's a
> fall, not someone sitting down."

*Hold up your phone.*

> "And that just arrived on the security guard's phone."

*Show the Telegram message.*

> "Look at the picture it sent. **Skeleton.** A guard who's never met him knows
> someone is down, which camera, which zone, and how long. They know nothing
> about who he is, and they don't need to.
>
> **CCTV is a witness. This is a first responder.**"

*B gets up.*

---

## 1:20 — The Signal for Help (25 seconds)

> **A:** "Here's the part we're proudest of. This system can't hear you, can't
> recognise you and will never know your name. So how does someone in trouble
> tell it anything?"

*B faces the camera, holds up a palm, tucks the thumb in, then folds the
fingers over it.*

> **A:** "That's the Signal for Help — the Canadian Women's Foundation gesture
> for *'I'm in danger and I can't say so out loud.'*"

*The alert appears — silently.*

> "Notice what didn't happen. **No chime. No flashing.** If someone makes that
> signal because of the person standing next to them, the last thing you want
> is a noise announcing it. It goes to the guard's phone and nowhere else.
>
> It's one gesture, and it's the only channel a person can deliberately open to
> a system that's promised not to watch them."

---

## 1:45 — The quiet ones (30 seconds)

*Move quickly. Point at the Observations column as you talk.*

> **A:** "Ten more rules are running on every camera, all the time."

*B puts a backpack down and walks away.*

> "Unattended bag — ten seconds in demo mode, thirty in reality. And it's not
> just 'a bag is still': somebody has to actually **stop** near it to count as
> taking it back, or in a busy corridor the timer would reset forever and the
> rule would quietly never fire."

*Point at the fire-exit zone on a tile.*

> "Fire exit blocked. Nobody is standing in a stairwell watching for the day
> someone leaves a crate there — this is already looking at the room, so it's
> free."

*Point at the counting line and the occupancy number.*

> "Entry and exit counting — how full is the library, did the hall actually
> clear during the drill. Crowd density. Running in the same direction, which
> is panic rather than one person late for a lecture. Loitering. After hours.
> And exam mode, which flags a phone visible at a desk — **for an invigilator
> to go and look at.** It can't tell you who's sitting there. That's deliberate."

---

## 2:15 — Ask the cameras (20 seconds)

*Type into the Ask box: "Summarise everything that happened today."*

> **A:** "You shouldn't need to scroll a log to run a building."

*Answer appears.*

> "That's Claude reading the event records. And read what it got sent —
> structured rows. Camera, time, rule, the measurements that triggered it.
> **No images. No clips. Nothing that could identify anyone leaves this
> machine.** One function builds that payload and you can read the whole thing
> in thirty seconds."

*Optionally: open an alert, click **Write the report**.*

> "And every incident writes its own report — what happened, which rule fired
> and why, the timeline around it, and what to do. Downloadable, for the
> incident file."

---

## 2:35 — The ledger (25 seconds) — *end here, always*

*Scroll to the privacy ledger. Let the big gold **0** sit on screen.*

> **A:** "This is the whole argument.
>
> **Zero hours of footage stored.** Not 'deleted after thirty days' — zero,
> since the day it was switched on. There is no code in this system capable of
> writing continuous video.
>
> What it does hold is on the left: fifteen seconds in memory that's
> overwritten continuously — cut the power and it's gone — and the handful of
> fifteen-second clips from actual incidents. Right now that's *(read the
> number)* clips. That's the total raw footage this system has ever produced."

*Point at the right column.*

> "Zero faces. Zero names. Zero re-identification. Not as policy — those things
> aren't in the code."

*Scroll to the audit list.*

> "And when staff genuinely need to see a face, they can. With the PIN."

*Click **Raw view** on a tile, enter `2026`. The raw camera appears.*

> "There we are. And —" *(point at the audit log, where the entry has already
> appeared)* "— that's now in the ledger. Time, camera, who asked. They can see
> faces when they need to. **They cannot do it quietly.**
>
> Normal CCTV records thousands of innocent students to catch one incident.
> This watches no one, until someone gives it a reason to."

*Lock the raw view again. Stop talking.*

---

## If something goes wrong

| Problem | What to do |
|---|---|
| Camera won't open | Switch to the simulated feed: it's already a tile, or restart with `--sim`. Say "that's a scripted scene, and it runs through the identical detectors — which is also how we test them." |
| Fall doesn't fire | Stay down. In demo mode it needs 1.2 s on the floor and a clean view of your hips. Lie across the frame, not towards the camera. |
| Nothing is detected at all | Step back — you need your whole body in frame. Check the tile says **live** and the fps is above 5. |
| Telegram silent | Skip it; the dashboard alert is the same information. Don't debug on stage. |
| Claude answers oddly | It's running the offline keyword search — say so. "No API key on this machine, so that's the local index. The point is it degrades instead of breaking." |
| Wifi dies entirely | Everything except Telegram and Claude keeps working. Detection, clips, dashboard are all local. |

---

## The three sentences to memorise

1. **"Normal CCTV records thousands of innocent students to catch one incident.
   This watches no one, until someone gives it a reason to."**
2. **"CCTV is a witness. This is a first responder."**
3. **"Zero hours of footage stored — not as a policy, but because there's no
   code in it that can."**
