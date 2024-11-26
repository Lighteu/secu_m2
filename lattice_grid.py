from manim import *
import numpy as np

class LatticeGridWithVectorCombinations(Scene):
    def construct(self):
        # Set up the coordinate system
        axes = NumberPlane(x_range=(-7, 7, 1), y_range=(-4, 4, 1))
        self.add(axes)

        # Define two base vectors
        v1 = np.array([1, 0, 0])  # Vector along x-axis
        angle = 60 * DEGREES
        v2 = np.array([np.cos(angle), np.sin(angle), 0])  # Vector at 60 degrees to x-axis

        # Draw the base vectors
        vector1 = Arrow(ORIGIN, v1, buff=0, color=RED)
        vector2 = Arrow(ORIGIN, v2, buff=0, color=BLUE)
        self.play(Create(vector1), Create(vector2))

        # Label the base vectors
        label1 = MathTex(r"\vec{v}_1").next_to(vector1.get_end(), UP)
        label2 = MathTex(r"\vec{v}_2").next_to(vector2.get_end(), UP)
        self.play(Write(label1), Write(label2))

        self.wait(1)

        # Define ranges for m and n to control the number of points
        m_range = range(-2, 3)
        n_range = range(-2, 3)

        # Animate the combination of vectors for each lattice point
        for m in m_range:
            for n in n_range:
                # Skip the origin to avoid redundant animation
                if m == 0 and n == 0:
                    continue

                # Calculate the lattice point
                point = m * v1 + n * v2

                # Create scaled vectors
                scaled_v1 = Arrow(ORIGIN, m * v1, buff=0, color=RED)
                scaled_v2 = Arrow(m * v1, point, buff=0, color=BLUE)

                # Create labels for the scaled vectors
                m_label = MathTex(f"{m}\\vec{{v}}_1").scale(0.6)
                n_label = MathTex(f"{n}\\vec{{v}}_2").scale(0.6)

                # Position labels appropriately
                m_label.next_to(scaled_v1, DOWN if m >= 0 else UP, buff=0.1)
                n_label.next_to(scaled_v2, LEFT if n >= 0 else RIGHT, buff=0.1)

                # Animate the vectors and labels pointing to the lattice point
                self.play(
                    Create(scaled_v1),
                    Write(m_label)
                )
                self.play(
                    Create(scaled_v2),
                    Write(n_label)
                )

                # Show the lattice point
                dot = Dot(point=point, radius=0.05, color=YELLOW)
                self.play(FadeIn(dot))

                self.wait(0.5)

                # Remove the vectors and labels before moving to the next point
                self.play(
                    FadeOut(scaled_v1),
                    FadeOut(scaled_v2),
                    FadeOut(m_label),
                    FadeOut(n_label),
                    FadeOut(dot)
                )

        self.wait(2)
