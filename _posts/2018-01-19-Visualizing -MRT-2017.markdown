---
layout: post
title: Visualizing 2017 MRT Performance using D3
date: 2018-01-19 13:32:20 +0300
description: This is my entry towards accenture visualization challenge and its my first time using javascript and d3. # Add post description (optional)
img: project-train-vis-2017/train_thumbnail.jpg # Add image post (optional)
tags: [D3, Visualization]
---

Philippines 2017 MRT Foot traffic Visualization made using D3. This was my entry for the Data Visualization category for Accenture's big data challenge. you can view the graphs [here](https://ryanliwag.github.io/datachallenge_vis_2018.github.io/)
This was also my first time using D3 and javascript in general. And its also my first shot at data visualization. I try to summarize in this post the process I have taken to build my first ever visualization using d3 and my shortcoming in what I could have done better. I definitely learned a lot, during the process and I think I can do much better next time a challenge like this comes around.

## Gathering the Data

So before any visualization is to occur, I needed to first gather the necessary data. Open data Philippines sadly only gives mrt data which is 2015 and below. So I opted to request data from eFOI Philippines and I asked for all data from 2015 till 2017. 2 weeks passed before I got the data, but what a fucking surprise, it's in excel format. The excel file was configured to their own style, and it was terrifying. Nonetheless, after some work I managed to get the mrt data into a csv format, additional data such as MRT issues, I managed to scrape them off MRT website. Followed by some pandas manipulation for data cleaning and handling missing data.
I have never done javascript in my life, so I figured this is a good chance to have a go at it. The main reason I wanted to try D3, is just because people said it was kind of hard to learn (I am a masochist, so let's GO). It took me a couple of days to get acclimated to javascript, but its a lot like python so I guess that's good.

## Building my first D3 Graphs

When It came to the visualizing the data, my first dumbass thought is how could I present as much data as possible into an interactive chart, that would also serve as a seemless UI that would hopefully be natural to interact with.

Key points I want to Visualize in my first Graph

* Project total entries and exits using a calendar heatmap z
* Turn the big calendar heat map into interactive buttons
* The interactive buttons will display bar charts that can show data on each station
* Then add extra buttons to toggle interesting events such as mrt issues/breakdowns or Holidays

After compiling what I want to visualize, I followed a typical color scheme to try and make it look passable. When visualizing the traffic throughout the year, I used a calendar heatmap to give a view of the total foot traffic. I don't know why, but during the time I thought it would be useful to add a side by side bar chart so that when you click on the calendar heat map it can give you a breakdown of the number of people entering and exiting a station. Hovering over the elements in the heatmap also gives additional details displayed by a tooltip.

![calendar_heatmap]({{site.baseurl}}/assets/img/project-train-vis-2017/train_calendar_heatmap_2.png)

The second graph I made, I wanted it to display what the average hourly traffic on each station. This was pretty simple, there is a simple slider whose value is pulled by a d3 function displaying the graph. The stopping motion of the slider to display values kind of irks me but I guess this will have to do since I also don't have a lot of time.

![hour_slider]({{site.baseurl}}/assets/img/project-train-vis-2017/train_hourly_vis.png)

## Shortcoming and Learnings

I learned a lot about how to use d3 ( such as chaining arguments and nesting data ). I am honestly not too happy with the results and it's already too late when I realized it. This visualization I made sadly doesn't tell any story. I was too focused on making it interactive, where interactivity might have been a mistake in the first place. Its cool and fancy, but it seems too much interactivity gives the user too much freedom and no clear direction to a story you want to tell. AHHHHHH, I should try harder the next time an opportunity to do visualization arrises.
