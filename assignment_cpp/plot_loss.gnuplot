# ================================
#  Training Loss Plot for SGD
# ================================

# Output settings (PNG image)
set terminal pngcairo size 1000,700 enhanced font "Arial,12"
set output "training_loss.png"

# Background
set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb "#f6f6f6" behind

# Grid
set grid xtics ytics mxtics mytics
set grid lw 1 lt 2 lc rgb "#cccccc"

# Labels and Title
set xlabel "Epoch"
set ylabel "Training Loss"
set title "Training Loss Curve" font "Arial,16"

# Line style
set style line 1 lc rgb "#0072c6" lw 2 lt 1  # Blue line, thick

# Optional: smooth the curve
# smooth csplines gives a nice smooth line (can disable if you want exact plot)
# plot "loss.dat" using 1:2 with lines smooth csplines ls 1 title "Loss"

# Plot (no smoothing)
plot "loss_history.txt" using 1 with lines ls 1 title "Loss"

# Done
