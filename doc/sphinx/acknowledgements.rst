.. highlight:: rest

Contributors and Acknowledgements
====================================

This is version 3 of slug, written primarily by Mark Krumholz. The code in this
repository is a ground-up reimplementation of slug version 2, so none of the code is
directly copied from that version. However, many people contributed ideas and assisted
with the development of slug versions 1 and 2 in ways that inevitably helped improve
the current version. Mark Krumholz acknowledges their contributions:

   * Michele Fumagalli: primary author of the slug version 2 test suite, co-author of version 1 of slug
   * Robert da Silva: primary author of version 1 of slug and of sfr_slug, wrote the first prototype version of slug2 and sfr_slug
   * Greg Ashworth: wrote the variable PDF and high-resolution UV modules in slug 2
   * Jonathan Parra: contributed code that became part of the slug_PDF module in slug 2
   * Teddy Rendahl: wrote the first version of cloudy_slug in slug 2
   * Michelle Myers: contributed to the development of cluster_slug in slug 2
   * Evan Demers: wrote the first version of the yield module in slug 2
   * Yusuke Fujimoto: helped debug callable library mode in slug 2, and wrote an interface between slug and enzo
   * Lucia Armillotta: contributed to the interface between slug 2 and gizmo
   * Ben Wibking: contributed to the interface between slug 2 and gizmo
   * Zipeng Hu: contributed to the interface between slug 2 and gizmo
   * Chuhan Zhang: contributed to the interface between slug 2 and gizmo
   * Alan (Ailun) Zhang did much of the preliminary work for the cloudy table-based treatment of nebular emission.
   * Alex Pedrini helped diagnose a rare failure mode in the tabulated integration code used to compute photometry in slug 2.

In addition to these direct contributors, we gratefully acknowledge the following people who provided some of the data on which either this version or earlier versions of slug relied:

   * Claus Leitherer's `starburst99 <http://www.stsci.edu/science/starburst99/docs/default.htm>`_ package was an inspiration, and the source of the track and atmosphere libraries used in versions 1 and 2.
   * The photometric filter library used in version 2 borrowed heavily from Charlie Conroy's `FSPS <https://code.google.com/p/fsps/>`_ package.
   * Daniela Calzetti provided the extinction curves.
   * Tuguldur Sukhbold provided core collapse supernova yield tables.
   * Amanda Karakas and Carolyn Doherty provided AGB yield tables.
