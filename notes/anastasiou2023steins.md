# anastasiou2023steins (VERIFIED: title, authors, year, venue, DOI via Crossref; abstract read. Body sections not read, so section contents are unchecked.)

**Claim**
A survey of how Stein operators, the linear operators behind Stein's method, have become tools in computational statistics. Per the abstract it covers benchmarking sampling methods such as approximate MCMC, deterministic alternatives to sampling, control variates, parameter estimation and goodness-of-fit testing.

**Method**
Review only, with no new experiments. Relevant parts for the study, by abstract and title only: Stein operators and discrepancies, KSD (goodness-of-fit and sampler benchmarking), kernel choice (IMQ kernel coverage unchecked), sliced KSD (unchecked; not in the abstract), and use in learning (parameter estimation; unchecked whether neural or representation learning is covered).

**Overlap with this study**
Background citation for KSD as a distributional regulariser against a Gaussian or non-Gaussian prior. Nothing on JEPA, temporal JEPA, autoencoders, nuisance, bottlenecks, identifiability or image data was found in the abstract (rest unchecked).

**Open gap**
The abstract does not mention KSD on embedding batches in self-supervised learning, comparison with SIGReg, VICReg or MMD, or bottleneck effects. The study could cover these on real images with known factors.

**BibTeX**
```
@article{Anastasiou_2023, title={Stein’s Method Meets Computational Statistics: A Review of Some Recent Developments}, volume={38}, ISSN={2168-8745}, url={http://dx.doi.org/10.1214/22-sts863}, DOI={10.1214/22-sts863}, number={1}, journal={Statistical Science}, publisher={Institute of Mathematical Statistics}, author={Anastasiou, Andreas and Barp, Alessandro and Briol, François-Xavier and Ebner, Bruno and Gaunt, Robert E. and Ghaderinezhad, Fatemeh and Gorham, Jackson and Gretton, Arthur and Ley, Christophe and Liu, Qiang and Mackey, Lester and Oates, Chris J. and Reinert, Gesine and Swan, Yvik}, year={2023}, month=Feb, pages={120–139} }
```
(arXiv:2105.03481 is the preprint; the arXiv record fetch was rate-limited.)

**Table row**
- regulariser: — (survey; KSD discussed)
- sliced/full: — (full KSD; sliced unchecked)
- prior: — (target density generic)
- nuisance: —
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: —
- objective: other (review)
- data: —
