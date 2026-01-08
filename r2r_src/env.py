"""Batched Room-to-Room navigation environment"""

import sys

sys.path.append("buildpy36")
sys.path.append("Matterport_Simulator/build/")
import MatterSim
import csv
import numpy as np
import math
import base64
import r2r_src.vln_utils as vln_utils
import json
import os
import random
import networkx as nx
from collections import defaultdict
from param import args

from r2r_src.vln_utils import load_datasets, load_nav_graphs, pad_instr_tokens

csv.field_size_limit(sys.maxsize)


class EnvBatch:
    """A simple wrapper for a batch of MatterSim environments,
    using discretized viewpoints and pretrained features"""

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:  # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = args.vfov
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print("The feature size is %d" % self.feature_size)
        else:
            print("    Image features not provided - in testing mode")
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = args.vfov
        self.sims = []
        # Use non-discretized mode when panoramic_horizontal_views != 12 to support custom angle increments
        use_discretized = args.panoramic_horizontal_views == 12
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            SCAN_DATA_DIR = "/home/iscas/Project/dataset/Matterport3D/data/"
            sim.setDatasetPath(SCAN_DATA_DIR)
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(use_discretized)
            # When discretized=True: Set increment/decrement to 30 degree. (otherwise by radians)
            # When discretized=False: makeAction uses radians for custom angle increments
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.initialize()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + "_" + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(
            zip(scanIds, viewpointIds, headings)
        ):
            # self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-(3*N-1)] (set only when viewing angles are discretized)
            [0-(N-1)] looking down, [N-(2*N-1)] looking at horizon, [2*N-(3*N-1)] looking up
            where N = panoramic_horizontal_views (default 12)
        :return: [ ((3*N, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        """Take an action using the full state dependent action interface (with batched input).
        Every action element should be an (index, heading, elevation) tuple."""
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


class R2RBatch:
    """Implements the Room to Room navigation task, using discretized viewpoints and pretrained features"""

    def __init__(
        self,
        feature_store,
        batch_size=100,
        seed=10,
        splits=["train"],
        tokenizer=None,
        name=None,
    ):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        else:
            self.feature_size = 2048
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for i_item, item in enumerate(load_datasets([split])):
                if args.test_only and i_item == 64:
                    break
                if (
                    split == "surrogate10"
                    or split == "val72"
                    or split == "MapGPT_30_scenes_processed_merged"
                    or split == "MapGPT_30_scenes_processed_merged_2"
                    or split == "NavGPT_30_scenes_processed_merged"
                    or split == "NavGPT2_30_scenes_processed_merged"
                    or split == "MapGPT_72_scenes_processed"
                    or split == "surrogate10_navgpt"
                    or split == "val72_navgpt"
                    or split == "surrogate10_navgpt2"
                    or split == "val72_navgpt2"
                ):
                    try:
                        new_item = dict(item)
                        new_item["instr_id"] = item["instr_id"]
                        new_item["instructions"] = [item["instruction"]]

                        """ BERT tokenizer """
                        instr_tokens = tokenizer.tokenize(item["instruction"])
                        padded_instr_tokens, num_words = pad_instr_tokens(
                            instr_tokens, args.maxInput
                        )
                        new_item["instr_encoding"] = tokenizer.convert_tokens_to_ids(
                            padded_instr_tokens
                        )

                        if (
                            new_item["instr_encoding"] is not None
                        ):  # Filter the wrong data
                            self.data.append(new_item)
                            scans.append(item["scan"])
                    except Exception as e:
                        print(e)

                        continue
                elif "/" in split:
                    try:
                        new_item = dict(item)
                        new_item["instr_id"] = item["path_id"]
                        new_item["instructions"] = item["instructions"][0]
                        new_item["instr_encoding"] = item["instr_enc"]
                        if (
                            new_item["instr_encoding"] is not None
                        ):  # Filter the wrong data
                            self.data.append(new_item)
                            scans.append(item["scan"])
                    except:
                        continue
                else:
                    # Split multiple instructions into separate entries
                    for j, instr in enumerate(item["instructions"]):
                        try:
                            new_item = dict(item)
                            new_item["instr_id"] = "%s_%d" % (item["path_id"], j)
                            new_item["instructions"] = instr

                            """ BERT tokenizer """
                            instr_tokens = tokenizer.tokenize(instr)
                            padded_instr_tokens, num_words = pad_instr_tokens(
                                instr_tokens, args.maxInput
                            )
                            new_item["instr_encoding"] = (
                                tokenizer.convert_tokens_to_ids(padded_instr_tokens)
                            )

                            if (
                                new_item["instr_encoding"] is not None
                            ):  # Filter the wrong data
                                self.data.append(new_item)
                                scans.append(item["scan"])
                        except:
                            continue

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = vln_utils.get_all_point_angle_feature()
        self.sim = vln_utils.new_simulator()
        self.buffered_state_dict = {}
        self.gt_trajs = self._get_gt_trajs(self.data)  # for evaluation
        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print(
            "R2RBatch loaded with %d instructions, using splits: %s"
            % (len(self.data), ",".join(splits))
        )

        # set the buffer to identify which vp has been added
        # self.instr_buffer = defaultdict(list)
        self.data_dict = {x["instr_id"]: x for x in self.data}

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print("Loading navigation graphs for %d scans" % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix : self.ix + batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[: self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def _next_minibatch_test(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.data[self.ix : self.ix + batch_size]
        if len(batch) < batch_size:
            # random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[: self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        """Reset the data index to beginning of epoch. Primarily for testing.
        You must still call reset() for a new episode."""
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def set_batch(self, scan_id="", viewpoint_id="", heading=0, **kwargs):
        """Load a new minibatch / episodes."""
        # ['sT4fr6TAbpF', 'ULsKaCPVFJR', 'JF19kD82Mey', 'sT4fr6TAbpF', 'S9hNv5qa7GM', 'cV4RVeZvu5T', 'p5wJjkQkbXX', 'vyrNrziPKCB']
        # ['6e41a7632c5a4048a17a316d7192b97e', '0b06e41938c84b91b41b6b14db0529b9', 'b98717151b7b49f59af95a9b7111a658', 'f39a5c37f80d48849b63bc84bbe395ae', '32fb55017460457cbe0b8d1790a54786', 'aa4cfd0126dd4c6a9c533ca9cb4a033d', '897ee6cdc5314ffaa23bf487ac2e0def', 'adddb681dbba4a138e296d6be545cc69']
        # [6.282, 4.534, 0.372, 6.271, 0.66, 3.139, 3.145, 6.281]

        self.batch = [self.data[1] for _ in range(self.batch_size)]
        scanIds = [scan_id] * self.batch_size
        viewpointIds = [viewpoint_id] * self.batch_size
        headings = [heading] * self.batch_size
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def _shortest_path_action(self, state, goalViewpointId):
        """Determine next action on the shortest path to goal, for supervised training."""
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId  # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading**2 + loc.rel_elevation**2)

        num_horizontal_views = args.panoramic_horizontal_views
        num_total_views = 3 * num_horizontal_views
        angle_increment = 360.0 / num_horizontal_views
        angle_increment_rad = math.radians(angle_increment)

        # Calculate base_heading based on viewId
        # This maintains the same relative heading logic as the original discrete code
        # The relative navigation angles should not change between discrete and continuous modes
        base_heading = (viewId % num_horizontal_views) * angle_increment_rad

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            # Check if we're using discretized mode (12 views) or non-discretized mode (custom views)
            use_discretized = args.panoramic_horizontal_views == 12

            for ix in range(num_total_views):
                if ix == 0:
                    # self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                    self.sim.newEpisode(
                        [scanId], [viewpointId], [0], [math.radians(-30)]
                    )
                elif ix % num_horizontal_views == 0:
                    # Move up one elevation level (30 degrees)
                    if use_discretized:
                        self.sim.makeAction([0], [1.0], [1.0])
                    else:
                        # Non-discretized mode: use radians (30 degrees = π/6)
                        self.sim.makeAction([0], [0.0], [math.radians(30)])
                else:
                    # Rotate horizontally
                    if use_discretized:
                        self.sim.makeAction([0], [1.0], [0])
                    else:
                        # Non-discretized mode: rotate by angle_increment_rad (e.g., 45 degrees = π/4)
                        self.sim.makeAction([0], [angle_increment_rad], [0])

                state = self.sim.getState()[0]
                # Note: When panoramic_horizontal_views != 12, MatterSim's viewIndex
                # (based on 30-degree increments) may not match our expected index ix.
                # We only assert when using the default 12 views.
                if args.panoramic_horizontal_views == 12:
                    assert (
                        state.viewIndex == ix
                    ), f"viewIndex mismatch: expected {ix}, got {state.viewIndex}"

                # Heading and elevation for the viewpoint center
                # Calculate relative heading: current heading minus base heading
                heading = (state.heading - base_heading) % (2 * math.pi)
                # print("heading", heading)
                # print("base_heading", base_heading)
                # print("state.heading", state.heading)
                elevation = state.elevation

                # Calculate pointId: in make_candidate, pointId should equal ix
                # because we start from newEpisode (heading=0) and rotate sequentially
                # For both discrete and continuous modes, pointId = ix
                point_id = ix

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = vln_utils.angle_feature(loc_heading, loc_elevation)
                    if (
                        loc.viewpointId not in adj_dict
                        or distance < adj_dict[loc.viewpointId]["distance"]
                    ):
                        adj_dict[loc.viewpointId] = {
                            "heading": loc_heading,
                            "elevation": loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            "scanId": scanId,
                            "viewpointId": loc.viewpointId,  # Next viewpoint id
                            "pointId": point_id,  # Use calculated pointId instead of ix
                            "distance": distance,
                            "idx": j + 1,
                            "feature": np.concatenate((visual_feat, angle_feat), -1),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {
                    key: c[key]
                    for key in [
                        "normalized_heading",
                        "normalized_elevation",
                        "elevation",
                        "scanId",
                        "viewpointId",
                        "pointId",
                        "idx",
                    ]
                }
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new["pointId"]
                normalized_heading = c_new["normalized_heading"]
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new["heading"] = loc_heading
                angle_feat = vln_utils.angle_feature(
                    c_new["heading"], c_new["elevation"]
                )
                c_new["feature"] = np.concatenate((visual_feat, angle_feat), -1)
                # NOTE: we need to use this
                # c_new.pop("normalized_heading")
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            # Calculate viewId for make_candidate and base_view_id
            # In discrete mode, use state.viewIndex
            # In continuous mode, calculate from heading and elevation
            if args.panoramic_horizontal_views == 12:
                # Discrete mode: state.viewIndex is available
                viewId = state.viewIndex
                base_view_id = state.viewIndex
            else:
                # Continuous mode: calculate current view position from heading and elevation
                num_horizontal_views = args.panoramic_horizontal_views
                angle_increment_rad = math.radians(360.0 / num_horizontal_views)

                # Determine elevation level
                if state.elevation < -0.2:
                    elev_level = 0
                elif state.elevation > 0.2:
                    elev_level = 2
                else:
                    elev_level = 1

                # Calculate horizontal index from heading
                heading_normalized = state.heading % (2 * math.pi)
                horiz_idx = (
                    int(round(heading_normalized / angle_increment_rad))
                    % num_horizontal_views
                )

                # Calculate pointId/viewId
                viewId = elev_level * num_horizontal_views + horiz_idx
                base_view_id = viewId

            if feature is None:
                num_total_views = 3 * args.panoramic_horizontal_views
                feature = np.zeros((num_total_views, 2048))

            # Full features
            candidate = self.make_candidate(
                feature, state.scanId, state.location.viewpointId, viewId
            )

            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            # get the index of current vp in the path
            # index_vp = item["path"].index(state.location.viewpointId)
            # index_vp_next = None
            # for index_vp, vp in enumerate(item["path"]):
            #     if (
            #         index_vp == state.location.viewpointId
            #         and index_vp not in self.instr_buffer[item["instr_id"]]
            #     ):
            #         self.instr_buffer[item["instr_id"]].append(index_vp)
            #         index_vp_next = (
            #             index_vp + 1
            #             if index_vp != len(item["path"]) - 1
            #             else len(item["path"]) - 1
            #         )
            #         break
            # assert index_vp_next is not None, "the viewpoint is not in the path!"
            obs.append(
                {
                    "instr_id": item["instr_id"],
                    "scan": state.scanId,
                    "viewpoint": state.location.viewpointId,
                    "viewIndex": viewId,  # Use calculated viewId for both discrete and continuous modes
                    "heading": state.heading,
                    "elevation": state.elevation,
                    "feature": feature,
                    "candidate": candidate,
                    "navigableLocations": state.navigableLocations,
                    "instructions": item["instructions"],
                    "teacher": self._shortest_path_action(state, item["path"][-1]),
                    # "teacher_baseline": index_vp_next,
                    "gt_path": item["path"],
                    "path_id": item["path_id"],
                }
            )
            if "instr_encoding" in item:
                obs[-1]["instr_encoding"] = item["instr_encoding"]
            # A2C reward. The negative distance between the state and the final state
            obs[-1]["distance"] = self.distances[state.scanId][
                state.location.viewpointId
            ][item["path"][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        """Load a new minibatch / episodes."""
        if batch is None:  # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:  # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[: len(batch)] = batch
            else:  # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item["scan"] for item in self.batch]
        viewpointIds = [item["path"][0] for item in self.batch]
        headings = [item["heading"] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def reset_test(self, **kwargs):
        """Load a new minibatch / episodes for test stage."""
        self._next_minibatch_test(**kwargs)

        scanIds = [item["scan"] for item in self.batch]
        viewpointIds = [item["path"][0] for item in self.batch]
        headings = [item["heading"] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        """Take action (same interface as makeActions)"""
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum["instructions"]))
            path += self.distances[datum["scan"]][datum["path"][0]][datum["path"][-1]]
        stats["length"] = length / len(self.data)
        stats["path"] = path / len(self.data)
        return stats

    def _get_gt_trajs(self, data):
        gt_trajs = {
            x["instr_id"]: (x["scan"], x["path"]) for x in data if len(x["path"]) >= 1
        }
        return gt_trajs

    def reset_to_starting_point(self):
        scanIds = [item["scan"] for item in self.batch]
        viewpointIds = [item["path"][0] for item in self.batch]
        headings = [item["heading"] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def get_scan_viewpoint_heading(self):
        scanIds = [item["scan"] for item in self.batch]
        viewpointIds = [item["path"][0] for item in self.batch]
        headings = [item["heading"] for item in self.batch]
        instr_ids = [item["instr_id"] for item in self.batch]
        return {
            "scanIds": scanIds,
            "viewpointIds": viewpointIds,
            "headings": headings,
            "instr_ids": instr_ids,
            "batch": self.batch,
        }

    # def set_scan_viewpoint_heading(self, location_tuple):
    #     scanIds = location_tuple["scanIds"]
    #     viewpointsIds = location_tuple["viewpointIds"]
    #     headings = location_tuple["headings"]
    #     self.batch = location_tuple["batch"]
    #     self.env.newEpisodes(scanIds, viewpointsIds, headings)
    #     return self._get_obs()

    def set_scan_viewpoint_heading(self, location_tuple):
        scanIds = location_tuple["scanIds"]
        viewpointsIds = location_tuple["viewpointIds"]
        headings = location_tuple["headings"]
        instr_ids = location_tuple["instr_ids"]
        self.env.newEpisodes(scanIds, viewpointsIds, headings)
        # self.batch = location_tuple["batch"]
        self.batch = [self.data_dict[x] for x in instr_ids]

        return self._get_obs()
