from manim import *

class DMLWEVisualization(Scene):
    def construct(self):
        # Step 1: Create a structured distribution (noisy lattice points)
        structured_points = VGroup(
            *[
                Dot(point=[x + 0.2 * (-1) ** i, y + 0.1 * (-1) ** j, 0], color=BLUE)
                for x in range(-5, 6)
                for y in range(-5, 6)
                for i, j in zip(range(2), range(2))
            ]
        )
        structured_label = Text("Structured Distribution (Noisy Lattice)", font_size=24).to_edge(UP)

        # Step 2: Create a random distribution
        import random
        random_points = VGroup(
            *[
                Dot(point=[random.uniform(-6, 6), random.uniform(-6, 6), 0], color=RED)
                for _ in range(100)
            ]
        )
        random_label = Text("Random Distribution", font_size=24).to_edge(UP)

        # Step 3: Show the structured distribution
        self.play(FadeIn(structured_points))
        self.play(Write(structured_label))
        self.wait(2)

        # Step 4: Transition to the random distribution
        self.play(FadeOut(structured_points), FadeOut(structured_label))
        self.play(FadeIn(random_points))
        self.play(Write(random_label))
        self.wait(2)

        # Step 5: Show the challenge of distinguishing them
        challenge_text = Text(
            "Can you tell which distribution is structured and which is random?",
            font_size=28,
            color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(challenge_text))

        self.wait(3)
