Think of it as the ultimate "super-balanced square grid" puzzle that mathematicians have been chasing for over 100 years.

Step 1: What's a Hadamard matrix? (The magic square we're hunting)

Imagine a square grid filled only with +1 and -1 (like pluses and minuses, or black/white tiles).

Rules for it to be "Hadamard" (the fanciest kind):



Every pair of rows must be perfectly balanced against each other.

Specifically: when you multiply matching positions (+1 × +1 = +1, -1 × -1 = +1, +1 × -1 = -1) and add them all up, you get 0 if the rows are different. (That's called "orthogonal" – like two arrows at right angles that don't lean toward each other.)

Same for columns.

It can only exist when the size n is 1, 2, or a multiple of 4.



Here's a tiny one (size 1 – boring but valid):

Just \[+1]

And the classic size 4 example – look how neatly the + and - are arranged:

en.wikipedia.orgsharetechnote.comsharetechnote.com





Step 2: Why are the rows "orthogonal"? (Visual analogy)

Picture each row as an arrow (vector) made of +1/-1 steps. Orthogonal means their dot product is zero – they're perfectly perpendicular, no agreement or disagreement bias.

Here's a nice visual of what orthogonal vectors look like (the right-angle idea):

bmanikan.medium.comximera.osu.edu



In Hadamard matrices, every pair of different rows has exactly half agreements and half disagreements – total balance!

Step 3: How do we build bigger ones? (The recursive trick)

There's a super cool doubling recipe (called the Sylvester construction) that lets you build the next size from the previous one.

Start with H₁ = \[+1]

Then H₂ = \[\[+ +]

\[+ -]]

Then H₄, H₈, H₁₆... you just keep wrapping the smaller one like this:

medium.comapieceofthepi.substack.com



This gives us Hadamard matrices for all powers of 2 (4, 8, 16, 32, 64... no problem).

Step 4: The big conjecture!

The Hadamard conjecture (from 1893) says:

"For every multiple of 4 (like 12, 20, 28, 36... up to huge numbers), there exists at least one Hadamard matrix of that size."

We already have them for 4, 8, 12, 16, 20... in fact, we've found them for every multiple of 4 we've checked – the current record is way up past 10,000! But nobody has proved it works for all of them forever.

It's one of the coolest open problems – like a treasure hunt where we've found tons of treasures but haven't proven the map covers the whole world.



---



think of these as fancy checkerboards where +1 is warm red and -1 is cool blue, perfectly balanced so no two rows (or columns) "agree" too much.

1\. Hadamard Matrix Order 4 (the classic small one via Sylvester construction)

This is the starting point beyond powers of 2 – super balanced!

2\. How we double up: Sylvester recursive construction

Start with a small Hadamard H, then build 2x bigger like this block pattern: H in top-left and top-right, H in bottom-left, -H in bottom-right. It magically stays perfectly orthogonal!

3\. Orthogonality analogy: Why rows are "perfectly perpendicular"

Each row is like an arrow – when they're orthogonal, their dot product is zero (no overlap in direction). Cos(90°) = 0, so total balance!

4\. A non-power-of-2 example: Order 12 (using Paley construction – unique up to flipping signs/rows)

Proof that it works for some multiples of 4 beyond powers of 2!

5\. The conjecture itself

We've found these super-balanced grids for every multiple of 4 we've checked (thousands!), but no proof they exist for ALL forever. It's like finding treasure everywhere on the map but not proving the whole world has it.



---





