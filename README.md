# Discrete Event Simulation: Hair Salon Analysis

A discrete event simulation project analyzing the performance of "m@ripuri" hair salon. The simulation models customer arrivals using a Poisson distribution (average 5 customers/hour) and service times following an exponential distribution (average 10 minutes/customer).

## Key Features
- Limited capacity system: 1 service chair + 4 waiting seats
- FIFO (First In, First Out) service policy
- Performance analysis including waiting times, queue length, and rejection probability
- Sensitivity analysis comparing current scenario vs. hiring an assistant

## Technologies
- Python
- Libraries: NumPy, Matplotlib, Pandas, Seaborn, SciPy

## Results
The simulation demonstrates that hiring an assistant would significantly improve service quality by reducing waiting times and rejection probability.