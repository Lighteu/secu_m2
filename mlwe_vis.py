from manim import *

class MLWEVisualization(Scene):
    def construct(self):
        # Step 1: Create a lattice grid
        lattice = NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            background_line_style={"stroke_color": BLUE, "stroke_width": 1},
        )
        self.add(lattice)

        # Step 2: Add some lattice points
        lattice_points = VGroup(
            *[Dot([x, y, 0], radius=0.08, color=WHITE) for x in range(-5, 6) for y in range(-5, 6)]
        )
        self.play(FadeIn(lattice_points))

        # Step 3: Visualize a secret vector
        secret_vector = Arrow(
            start=[0, 0, 0],
            end=[3, 2, 0],
            buff=0,
            color=GREEN,
            stroke_width=5,
        )
        self.play(Create(secret_vector))
        self.play(Write(Text("Secret Vector", font_size=24).next_to(secret_vector, UP)))

        # Step 4: Add noise to the vector
        noisy_point = Dot([3.5, 2.7, 0], radius=0.1, color=RED)
        noise_arrow = Arrow(
            start=[3, 2, 0],
            end=[3.5, 2.7, 0],
            buff=0,
            color=YELLOW,
            stroke_width=3,
        )
        self.play(Create(noisy_point), Create(noise_arrow))
        self.play(Write(Text("Noise", font_size=24).next_to(noise_arrow, RIGHT)))

        # Step 5: Show the noisy vector as the observed output
        noisy_vector_arrow = Arrow(
            start=[0, 0, 0],
            end=[3.5, 2.7, 0],
            buff=0,
            color=RED,
            stroke_width=5,
        )
        self.play(Transform(secret_vector, noisy_vector_arrow))
        self.play(Write(Text("Observed Vector", font_size=24).next_to(noisy_vector_arrow, UP)))

        # Step 6: Highlight the difficulty of recovering the secret
        difficulty_text = Text(
            "How to recover the secret vector from noisy observations?",
            font_size=28,
            color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(difficulty_text))

        self.wait(3)
