---
layout: page
permalink: /works/
title: research & projects
description: 
nav: true
nav_order: 1
---
<!-- _bibliography/papers -->

<h2> publications </h2>
<div class="publications">
{% bibliography -f papers -q @*[displaytype=paper]* %}
</div>

<!-- _bibliography/preprints -->
<h2> preprints </h2>
<div class="publications">
{% bibliography -f papers -q @*[displaytype=preprint]* %}
</div>