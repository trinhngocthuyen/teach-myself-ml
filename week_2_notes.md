Week 2
================

### Theory

- There are 2 major approaches towards linear regression: **Normal equation** & **Gradient descent**
- The 2 approaches have its own pros and cons, and suitable for a specific context:
	- Gradient descent:
		+ The learning process somehow depends on the learning rate (`alpha`)
		+ Iteration is needed to determine the point of convergence
		+ Works well even if `n` is large
	- Normal equation:
		+ No need to choose `alpha` (learning rate)
		+ Iteration is not needed (one single formula to solve them all)
		+ Feature scaling is not needed
		+ Not work well if `n` is large (approximately over 10.000) due to the expensive cost of determining the matrix inverse	
- Feature scaling:
	- Helps speed up learning process with gradient descent. If using large range of values, the decrease of the cost function will be relatively significant
	- Note: no need to use feature scaling for normal equation
- Others:
	- When working with gradient descent, should guarantee if it works correctly:
		+ Make a plot: `cost_function = f(nIterations)`, check whether the visual curve demonstrates a convergence (level-off)
		+ If not, you should adjust the learning rate

### Programming
#### Basic operators
- suppress output by semicolons
- `eye(4)`		--> Identity matrix
- `size(A)`		--> (nRows, nColumns)
- `length(A)`	--> max(nRows, nColumns)
- `who` 		--> List variables in the current scope
- `whos` 		--> who (with details)
- `save`		--> Save data to file
- `A = [A, extra_matrix]`		--> Concat extra_matrix to the right
- `A = [A; extra_matrix]`		--> Concat extra_matrix to the bottom
- `A(:)`		--> Put everything into a single vector
- `A .* B` 		--> C(i, j) = A(i, j) * B(i, j)
- `[value, idx] = max(A)`	--> max value & index by column
- `find(A < 5)`				--> Find elements less than 5 (then align them in a row vector)
- `[r, c] = find(A < 5)`	--> Similar to find(A < 5), but the return the index of (rol, col)
- `magic(N)`		--> Matrix with sum(col_i) = sum(col_j) = sum(row_p) = sum(row_q)
- `prod(A)`		--> Product of elements by column
- `flipud(A)`		--> Vertically flip the matrix

#### Plotting
- `hold on` 		--> Hold the current plot, we will draw the next plot in the same figure (with this plot)
	+ `plot(t, y1, 'b'); hold on; plot(t, y2, 'r');`
- `xlabel('time'); ylabel('distance');`
- `lgend('sin', 'cos')`
- `print -dpng 'myPlot.png'`	--> Save the plot to an image
- `subplot`
	+ `subplot(1, 2, 1); plot(t, y1);`	--> Move to the 1st cell of a (1 x 2) grid, the show the plot
	+ `subplot(1, 2, 2); plot(t, y2)`	--> Move to the 2st cell of a (1 x 2) grid, the show the plot
	+ `axis([0.5 1 -1 1])`				--> Scale: x in [0.5 -> -1], y in [-1 -> 1]
- `imagesc` <-- Advanced, no need to memorize it now

#### Others
- Use *vectorization* instead of loop b/c Matlab/Octave brings some efficient computations optimized by Linear Algebra theories