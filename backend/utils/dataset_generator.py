"""Synthetic dataset generator for mental health risk detection training."""

import csv
import os
import random

from config import DATASET_PATH, DATA_DIR


# Sample text templates for each risk level
LOW_RISK_TEXTS = [
    "Had a great day at work today. Feeling productive and accomplished!",
    "Just finished a wonderful dinner with friends. Life is good.",
    "Enjoying the beautiful weather outside. Going for a walk in the park.",
    "Started reading a new book and I'm really enjoying it so far.",
    "Had a productive meeting today. Things are moving in the right direction.",
    "Spent quality time with family this weekend. Feeling grateful.",
    "Just completed my morning workout. Feeling energized and ready for the day.",
    "Trying out a new recipe tonight. Cooking always makes me happy.",
    "Watched a great movie with friends. Sometimes simple things bring joy.",
    "Finished a big project at work. Feeling relieved and proud.",
    "Beautiful sunrise this morning. Starting the day with positivity.",
    "Had an amazing lunch with colleagues. Good conversations always lift my mood.",
    "Learning a new skill and making progress. Small wins matter!",
    "Grateful for the support system I have. Life has its ups and downs but I'm okay.",
    "Just came back from vacation feeling refreshed and recharged.",
    "Organized my room today and it feels so satisfying.",
    "Got a compliment at work today. Made my whole day brighter.",
    "Enjoying some quiet time alone. Self care is important.",
    "Had a fun game night with friends. Laughter is the best medicine.",
    "Feeling content with where I am in life right now.",
    "The sunset today was absolutely breathtaking. Nature is healing.",
    "Adopted a new puppy and I'm over the moon with happiness.",
    "Just ran my first 5K! Proud of myself for sticking with training.",
    "Coffee with an old friend today. Reconnecting feels wonderful.",
    "My garden is blooming and it brings me such peace.",
    "Volunteered at the shelter today. Giving back feels meaningful.",
    "Had a good conversation with my therapist. Feeling hopeful about progress.",
    "Finally got the promotion I've been working toward. Hard work pays off!",
    "Spent the day at the beach. The ocean always calms my mind.",
    "Listened to my favorite album today. Music is good for the soul.",
    "Just baked cookies for my neighbors. Spreading kindness feels good.",
    "Woke up feeling well-rested for the first time in a while. Simple pleasures.",
    "My team won the game tonight! Exciting times with good friends.",
    "Journaling has been helping me stay mindful and present.",
    "Today I choose happiness and gratitude for what I have.",
    "Love spending weekends with my family, just doing simple things together.",
    "Discovered a new hiking trail today. Adventure awaits!",
    "Feeling blessed to have such supportive coworkers.",
    "Made someone smile today and that made my day too.",
    "Enjoying a peaceful evening with a good book and some tea.",
    "Finally finished decorating my apartment. It really feels like home now.",
    "Had a fun day exploring the local farmers market.",
    "Meditation has been helping me stay centered lately.",
    "So proud of my friend who graduated today! Celebrating achievements is wonderful.",
    "Got creative and painted something today. Art is therapeutic.",
    "Today was a normal day and that's perfectly fine with me.",
    "Caught up on sleep this weekend and feeling much better now.",
    "Tried a new restaurant and the food was amazing!",
    "Feeling optimistic about the future and the possibilities ahead.",
    "Simple walk in the neighborhood reminded me to appreciate the little things.",
]

MEDIUM_RISK_TEXTS = [
    "Feeling a bit down today. Work has been really stressful lately.",
    "Can't seem to shake this feeling of worry. Everything feels uncertain.",
    "Haven't been sleeping well. My mind keeps racing at night.",
    "Feeling overwhelmed with responsibilities. Not sure how to manage everything.",
    "I keep comparing myself to others and it makes me feel inadequate.",
    "Things have been tough lately. Struggling to find motivation to do anything.",
    "Feeling disconnected from friends. Everyone seems to be moving on without me.",
    "Anxiety has been getting worse. Social situations make me really uncomfortable.",
    "I've been eating a lot more than usual. Food is my only comfort right now.",
    "Feeling exhausted all the time even though I haven't done much.",
    "Work stress is getting to me. I feel like I'm always behind.",
    "Having trouble concentrating. My thoughts keep wandering to negative places.",
    "I don't enjoy the things I used to. Everything feels flat and boring.",
    "Feeling lonely even when I'm around people. No one truly understands.",
    "I've been crying for no apparent reason. Not sure what's wrong with me.",
    "Can't stop overthinking about my past mistakes. The guilt is overwhelming.",
    "My relationships feel strained. I don't know how to communicate what I feel.",
    "I'm always tired but can't sleep. This cycle is really frustrating.",
    "Feeling like I'm not good enough for anything. Low confidence days.",
    "Some days are okay but most feel gray and empty.",
    "I keep putting things off because nothing feels worth the effort.",
    "Had a panic attack today. It came out of nowhere and scared me.",
    "I'm struggling with change. Everything new makes me anxious.",
    "Feeling stuck in life. Like I'm not making any progress at all.",
    "I've been isolating myself more and more. People drain my energy.",
    "The pressure to succeed is crushing me. I feel like a failure.",
    "I can't remember the last time I genuinely laughed or felt happy.",
    "My mood swings are getting worse. One minute I'm fine, the next I'm not.",
    "I feel like a burden to my friends and family. They'd be better off.",
    "Constant headaches from stress. My body is telling me something is wrong.",
    "Struggling to get out of bed most mornings. What's the point anyway.",
    "Lost interest in hobbies I used to love. Nothing excites me anymore.",
    "I feel numb most of the time. Not sad exactly just empty.",
    "Everything feels like too much effort lately. Even basic tasks are hard.",
    "I keep replaying conversations in my head wondering what I did wrong.",
    "Feeling invisible at work and in my personal life.",
    "My anxiety is making it hard to eat. I've lost my appetite completely.",
    "I wish I could just press pause on life for a while.",
    "Feel like I'm walking through fog. Nothing seems clear or real.",
    "Started avoiding phone calls and messages. Social interaction feels exhausting.",
    "I'm always on edge waiting for something bad to happen.",
    "Can't focus on anything productive. My brain feels scattered.",
    "Feeling defeated after another rejection. Starting to lose hope.",
    "I pretend to be fine around others but inside I'm struggling.",
    "The weekend used to be something I looked forward to. Now it's just more emptiness.",
    "I've been having bad dreams that leave me feeling unsettled all day.",
    "Constant worry about money and the future. It's consuming me.",
    "I feel like I'm falling behind everyone else my age.",
    "Sometimes I wonder if anyone would notice if I just disappeared for a while.",
    "I'm tired of pretending everything is okay when it really isn't.",
]

HIGH_RISK_TEXTS = [
    "I don't see the point in living anymore. Everything feels hopeless and dark.",
    "I wish I could just disappear forever. Nobody would even care.",
    "I've been thinking about hurting myself. The pain inside is unbearable.",
    "Life is meaningless. I'm just existing, not living. I want it to stop.",
    "I feel completely empty inside. Nothing brings me joy or hope.",
    "I can't take this pain anymore. Every day is a struggle to survive.",
    "Sometimes I think everyone would be better off without me in their lives.",
    "I've been having thoughts about ending it all. The darkness is consuming me.",
    "I hate myself and everything about my life. There's no way out.",
    "The world would be better without me. I'm nothing but a burden.",
    "I can't stop crying. The sadness is so deep it physically hurts.",
    "I've been self-harming to cope with the emotional pain. It's the only relief.",
    "I don't want to wake up tomorrow. Sleep is the only escape.",
    "I feel trapped in a life I never wanted. There's no escape from this misery.",
    "I've lost all hope. Nothing is ever going to get better. Why try.",
    "I've been planning how to end my suffering. I can't do this anymore.",
    "No one cares about me. I'm completely alone in this dark world.",
    "I've written goodbye letters. I don't think I can hold on much longer.",
    "Every day is torture. I don't know how much more I can take.",
    "I feel like I'm drowning and no one can hear me screaming for help.",
    "I've stopped eating and caring about myself. What's the point of anything.",
    "The voices in my head keep telling me I'm worthless and should give up.",
    "I fantasize about not existing. The world would keep spinning without me.",
    "I've been drinking heavily to numb the pain. Nothing else works anymore.",
    "I can't remember what happiness feels like. It's been dark for so long.",
    "I've pushed everyone away. I'm completely isolated and broken beyond repair.",
    "I want the pain to stop. I just want everything to stop permanently.",
    "I feel dead inside already. My body is just going through the motions.",
    "I've been researching methods to end my life. I'm scared of myself.",
    "There is no future for me. Everything I touch turns to ash.",
    "I'm a complete failure who doesn't deserve to live.",
    "I can't breathe through the panic and despair. It never stops.",
    "I've given away my possessions. I won't need them where I'm going.",
    "The darkness has won. I have no strength left to fight it.",
    "I wake up disappointed that I woke up at all.",
    "My existence is meaningless suffering. I'm tired of pretending otherwise.",
    "I've been cutting myself more frequently. The scars are getting deeper.",
    "No one in this world would miss me. I'm invisible and insignificant.",
    "I've stopped taking my medication because I don't care what happens to me.",
    "Death seems like the only peaceful option left for someone like me.",
    "I've been isolating for weeks. I haven't left my room or spoken to anyone.",
    "I wrote my final note today. I'm so tired of fighting this battle alone.",
    "Everything is crumbling around me and I have no will to rebuild.",
    "I hate waking up every morning to face another day of this nightmare.",
    "I'm a mistake that should never have been born into this world.",
    "The suicidal thoughts are getting louder and harder to ignore.",
    "I've lost everything that mattered. There's nothing left holding me here.",
    "I can't function anymore. Basic existence is an overwhelming burden.",
    "I'm beyond help. No therapy or medication can fix what's broken inside me.",
    "This is my last cry for help. If no one hears me I'm done trying.",
]


def generate_dataset(n_samples_per_class: int = 500, output_path: str = None) -> str:
    """Generate a synthetic mental health dataset with augmented variations."""
    if output_path is None:
        output_path = DATASET_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    augmentation_templates = {
        "prefix": [
            "", "honestly ", "tbh ", "feeling like ", "i think ",
            "today i feel ", "right now ", "lately ", "been thinking ",
            "can't help but feel ", "just wanted to say ",
        ],
        "suffix": [
            "", " honestly", " i guess", " not sure what to do",
            " it is what it is", " anyone else feel this way",
            " need to talk about this", " just venting",
        ],
    }

    rows = []

    for label, templates, risk_code in [
        ("Low", LOW_RISK_TEXTS, 0),
        ("Medium", MEDIUM_RISK_TEXTS, 1),
        ("High", HIGH_RISK_TEXTS, 2),
    ]:
        generated = 0
        while generated < n_samples_per_class:
            base_text = random.choice(templates)
            prefix = random.choice(augmentation_templates["prefix"])
            suffix = random.choice(augmentation_templates["suffix"])

            text = f"{prefix}{base_text}{suffix}".strip()

            # Random word-level augmentation
            if random.random() < 0.3:
                words = text.split()
                if len(words) > 5:
                    # Randomly duplicate or drop a word
                    idx = random.randint(0, len(words) - 1)
                    if random.random() < 0.5:
                        words.insert(idx, words[idx])
                    else:
                        words.pop(idx)
                    text = " ".join(words)

            rows.append({"text": text, "label": label, "risk_code": risk_code})
            generated += 1

    # Shuffle the dataset
    random.seed(42)
    random.shuffle(rows)

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "risk_code"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {len(rows)} samples → {output_path}")
    print(f"   Low: {n_samples_per_class} | Medium: {n_samples_per_class} | High: {n_samples_per_class}")

    return output_path


if __name__ == "__main__":
    generate_dataset()
