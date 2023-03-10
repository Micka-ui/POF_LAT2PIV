<h1> Physics-informed neural networks for gravity currents reconstruction from limited data </h1>

<p> The present work investigates the use of physics-informed neural networks (PINNs) \cite{Rassi}for the three-dimensional (3D)
reconstruction of unsteady gravity currents from limited data. In the PINN context, the flow fields are reconstructed
by training a neural network whose objective function penalizes the mismatch between the network predictions and the
observed data and embeds the underlying equations using automatic differentiation. This study relies on a high-fidelity
numerical experiment of the canonical lock-exchange configuration. This allows us to benchmark quantitatively the
PINNs reconstruction capabilities on several training databases that mimic state-of-the-art experimental measurement
techniques for density and velocity. Notably, spatially averaged density measurements by light attenuation technique
(LAT) are employed for the training procedure. We propose an experimental setup that combines density measurement
by LAT and two independent planar velocity measurements by particle image velocimetry (PIV). The so-called LAT-
2PIV setup gives the most promising results for flow reconstruction by PINNs, with respect to its accuracy and cost
efficiency.</p>

## CITATION 

      @article{3DInferred,
         author = {Delcey,Mickael  and Cheny,Yoann  and Kiesgen de Richter,Sébastien },
         title = {Physics-informed neural networks for gravity currents reconstruction from limited data},
         journal = {Physics of Fluids},
         }
