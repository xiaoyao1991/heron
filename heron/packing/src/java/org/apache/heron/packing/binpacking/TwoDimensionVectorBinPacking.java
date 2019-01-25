/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.heron.packing.binpacking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

import org.apache.heron.api.generated.TopologyAPI;
import org.apache.heron.api.utils.TopologyUtils;
import org.apache.heron.common.basics.ByteAmount;
import org.apache.heron.packing.builder.Container;
import org.apache.heron.packing.builder.ContainerIdScorer;
import org.apache.heron.packing.builder.HomogeneityScorer;
import org.apache.heron.packing.builder.InstanceCountScorer;
import org.apache.heron.packing.builder.PackingPlanBuilder;
import org.apache.heron.packing.builder.ResourceRequirement;
import org.apache.heron.packing.builder.Scorer;
import org.apache.heron.packing.builder.SortingStrategy;
import org.apache.heron.packing.exceptions.ResourceExceededException;
import org.apache.heron.packing.utils.PackingUtils;
import org.apache.heron.spi.common.Config;
import org.apache.heron.spi.common.Context;
import org.apache.heron.spi.packing.IPacking;
import org.apache.heron.spi.packing.IRepacking;
import org.apache.heron.spi.packing.PackingException;
import org.apache.heron.spi.packing.PackingPlan;
import org.apache.heron.spi.packing.Resource;

import static org.apache.heron.api.Config.TOPOLOGY_CONTAINER_CPU_PADDING;
import static org.apache.heron.api.Config.TOPOLOGY_CONTAINER_MAX_CPU_HINT;
import static org.apache.heron.api.Config.TOPOLOGY_CONTAINER_MAX_DISK_HINT;
import static org.apache.heron.api.Config.TOPOLOGY_CONTAINER_MAX_NUM_INSTANCES;
import static org.apache.heron.api.Config.TOPOLOGY_CONTAINER_MAX_RAM_HINT;
import static org.apache.heron.api.Config.TOPOLOGY_CONTAINER_PADDING_PERCENTAGE;
import static org.apache.heron.api.Config.TOPOLOGY_CONTAINER_RAM_PADDING;

/**
 * FirstFitDecreasing packing algorithm
 * <p>
 * This IPacking implementation generates a PackingPlan based on the
 * First Fit Decreasing heuristic for the binpacking problem. The algorithm attempts to minimize
 * the amount of resources used.
 * <p>
 * Following semantics are guaranteed:
 * 1. Supports heterogeneous containers. The number of containers used is determined
 * by the algorithm and not by the config file.
 * The user provides a hint for the maximum container size and a padding percentage.
 * The padding percentage whose values range from [0, 100], determines the per container
 * resources allocated for system-related processes (e.g., the stream manager).
 * <p>
 * 2. The user provides a hint for the maximum CPU, RAM and Disk that can be used by each container through
 * the org.apache.heron.api.Config.TOPOLOGY_CONTAINER_MAX_CPU_HINT,
 * org.apache.heron.api.Config.TOPOLOGY_CONTAINER_MAX_RAM_HINT,
 * org.apache.heron.api.Config.TOPOLOGY_CONTAINER_MAX_DISK_HINT parameters.
 * If the parameters are not specified then a default value is used for the maximum container
 * size.
 * <p>
 * 3. The user provides a percentage of each container size that will be used for padding
 * through the org.apache.heron.api.Config.TOPOLOGY_CONTAINER_PADDING_PERCENTAGE
 * If the parameter is not specified then a default value of 10 is used (10% of the container size)
 * <p>
 * 4. The RAM required for one instance is calculated as:
 * value in org.apache.heron.api.Config.TOPOLOGY_COMPONENT_RAMMAP if exists, otherwise,
 * the default RAM value for one instance.
 * <p>
 * 5. The CPU required for one instance is calculated as the default CPU value for one instance.
 * <p>
 * 6. The disk required for one instance is calculated as the default disk value for one instance.
 * <p>
 * 7. The RAM required for a container is calculated as:
 * (RAM for instances in container) + (paddingPercentage * RAM for instances in container)
 * <p>
 * 8. The CPU required for a container is calculated as:
 * (CPU for instances in container) + (paddingPercentage * CPU for instances in container)
 * <p>
 * 9. The disk required for a container is calculated as:
 * (disk for instances in container) + ((paddingPercentage * disk for instances in container)
 * <p>
 * 10. The pack() return null if PackingPlan fails to pass the safe check, for instance,
 * the size of RAM for an instance is less than the minimal required value.
 */
public class TwoDimensionVectorBinPacking implements IPacking, IRepacking {

  private static final Logger LOG = Logger.getLogger(TwoDimensionVectorBinPacking.class.getName());

  // default
  private static final int DEFAULT_CONTAINER_PADDING_PERCENTAGE = 10;
  private static final ByteAmount DEFAULT_CONTAINER_RAM_PADDING = ByteAmount.fromGigabytes(1);
  private static final double DEFAULT_CONTAINER_CPU_PADDING = 1.0;
  private static final int DEFAULT_MAX_NUM_INSTANCES_PER_CONTAINER = 4;

  private TopologyAPI.Topology topology;

  // instance & container
  private Resource defaultInstanceResources;
  private Resource maxContainerResources;
  private int maxNumInstancesPerContainer;

  // padding
  private Resource padding;

  private int numContainers = 0;

  @Override
  public void initialize(Config config, TopologyAPI.Topology inputTopology) {
    this.topology = inputTopology;
    setPackingConfigs(config);
    LOG.info(String.format("Initalizing TwoDimensionVectorBinPacking: \n"
        + "Max number of instances per container: %d \n"
        + "Default instance CPU: %f, Default instance RAM: %s, Default instance DISK: %s \n"
        + "Paddng: %s \n"
        + "Container CPU: %f, Container RAM: %s, Container DISK: %s",
        this.maxNumInstancesPerContainer,
        this.defaultInstanceResources.getCpu(),
        this.defaultInstanceResources.getRam().toString(),
        this.defaultInstanceResources.getDisk().toString(),
        this.padding.toString(),
        this.maxContainerResources.getCpu(),
        this.maxContainerResources.getRam().toString(),
        this.maxContainerResources.getDisk().toString()));
  }

  /**
   * Instantiate the packing algorithm parameters related to this topology.
   */
  private void setPackingConfigs(Config config) {
    List<TopologyAPI.Config.KeyValue> topologyConfig = topology.getTopologyConfig().getKvsList();

    // instance default resources are acquired from heron system level config
    this.defaultInstanceResources = new Resource(
        Context.instanceCpu(config),
        Context.instanceRam(config),
        Context.instanceDisk(config));

    int paddingPercentage = TopologyUtils.getConfigWithDefault(topologyConfig,
        TOPOLOGY_CONTAINER_PADDING_PERCENTAGE, DEFAULT_CONTAINER_PADDING_PERCENTAGE);
    ByteAmount ramPadding = TopologyUtils.getConfigWithDefault(topologyConfig,
        TOPOLOGY_CONTAINER_RAM_PADDING, DEFAULT_CONTAINER_RAM_PADDING);
    double cpuPadding = TopologyUtils.getConfigWithDefault(topologyConfig,
        TOPOLOGY_CONTAINER_CPU_PADDING, DEFAULT_CONTAINER_CPU_PADDING);

    this.maxNumInstancesPerContainer = TopologyUtils.getConfigWithDefault(topologyConfig,
        TOPOLOGY_CONTAINER_MAX_NUM_INSTANCES, DEFAULT_MAX_NUM_INSTANCES_PER_CONTAINER);

    // container default resources are computed as:
    // max number of instances per container * default instance resources
    double containerDefaultCpu = this.defaultInstanceResources.getCpu()
        * maxNumInstancesPerContainer;
    ByteAmount containerDefaultRam = this.defaultInstanceResources.getRam()
        .multiply(maxNumInstancesPerContainer);
    ByteAmount containerDefaultDisk = this.defaultInstanceResources.getDisk()
        .multiply(maxNumInstancesPerContainer);

    double containerCpuWithoutPadding = TopologyUtils.getConfigWithDefault(topologyConfig,
        TOPOLOGY_CONTAINER_MAX_CPU_HINT, containerDefaultCpu);
    ByteAmount containerRamWithoutPadding = TopologyUtils.getConfigWithDefault(topologyConfig,
        TOPOLOGY_CONTAINER_MAX_RAM_HINT, containerDefaultRam);
    ByteAmount containerDiskWithoutPadding = TopologyUtils.getConfigWithDefault(topologyConfig,
        TOPOLOGY_CONTAINER_MAX_DISK_HINT, containerDefaultDisk);

    // finalize padding
    cpuPadding = Math.max(cpuPadding, containerCpuWithoutPadding * paddingPercentage);
    ramPadding = ByteAmount.fromBytes(
        Math.max(ramPadding.asBytes(), containerRamWithoutPadding.asBytes()));
    this.padding = new Resource(cpuPadding, ramPadding, ByteAmount.ZERO);

    // finalize container resources
    this.maxContainerResources = new Resource(containerCpuWithoutPadding + cpuPadding,
        containerRamWithoutPadding.plus(ramPadding),
        containerDiskWithoutPadding.increaseBy(paddingPercentage));
  }

  private PackingPlanBuilder newPackingPlanBuilder(PackingPlan existingPackingPlan) {
    return new PackingPlanBuilder(topology.getId(), existingPackingPlan)
        .setMaxContainerResource(maxContainerResources)
//        .setDefaultInstanceResource(defaultInstanceResources)
        .setRequestedContainerPadding(padding);
//        .setRequestedComponentRam(TopologyUtils.getComponentRamMapConfig(topology))
//        .setRequestedComponentCpu(TopologyUtils.getComponentCpuMapConfig(topology));
  }

  /**
   * Get a packing plan using First Fit Decreasing
   *
   * @return packing plan
   */
  @Override
  public PackingPlan pack() {
    PackingPlanBuilder ramFirstPlanBuilder = newPackingPlanBuilder(null);
    PackingPlanBuilder cpuFirstPlanBuilder = newPackingPlanBuilder(null);

    // Get the instances using FFD allocation
    try {
      ramFirstPlanBuilder = get2DVectorBinAllocation(ramFirstPlanBuilder,
          SortingStrategy.RAM_FIRST);
      cpuFirstPlanBuilder = get2DVectorBinAllocation(cpuFirstPlanBuilder,
          SortingStrategy.CPU_FIRST);

      PackingPlan ramFirstPlan = ramFirstPlanBuilder.build();
      PackingPlan cpuFirstPlan = cpuFirstPlanBuilder.build();

      return pickBetter(ramFirstPlan, cpuFirstPlan);
    } catch (ResourceExceededException e) {
      throw new PackingException("Could not allocate all instances to packing plan", e);
    }
  }

  /**
   * Pick the better plan by checking which plan dominates in terms of:
   * overall resource utility, average container resource utility,
   * container ram utility stdev, and container cpu utility stdev.
   * @param plan1
   * @param plan2
   * @return the better plan
   */
  private PackingPlan pickBetter(PackingPlan plan1, PackingPlan plan2) {
    int plan1Points = 0;
    int plan2Points = 0;

    PackingPlan.ResourceUtility plan1ResourceUtility = plan1.getPackingPlanResourceUtility();
    PackingPlan.ResourceUtility plan2ResourceUtility = plan2.getPackingPlanResourceUtility();
    if (plan1ResourceUtility.compare(plan2ResourceUtility) > 0) {
      plan1Points++;
    } else if (plan1ResourceUtility.compare(plan2ResourceUtility) < 0) {
      plan2Points++;
    }

    PackingPlan.ResourceUtility plan1AvgContainerResourceUtility
        = plan1.getAvgContainerResourceUtility();
    PackingPlan.ResourceUtility plan2AvgContainerResourceUtility
        = plan2.getAvgContainerResourceUtility();
    if (plan1AvgContainerResourceUtility.compare(plan2AvgContainerResourceUtility) > 0) {
      plan1Points++;
    } else if (plan1AvgContainerResourceUtility.compare(plan2AvgContainerResourceUtility) < 0) {
      plan2Points++;
    }

    double plan1ContainerRamUtilityStdev = plan1.getContainerRamUtilityStdev();
    double plan2ContainerRamUtilityStdev = plan2.getContainerRamUtilityStdev();
    if (Double.compare(plan1ContainerRamUtilityStdev, plan2ContainerRamUtilityStdev) > 0) {
      plan1Points++;
    } else if (Double.compare(plan1ContainerRamUtilityStdev, plan2ContainerRamUtilityStdev) < 0) {
      plan2Points++;
    }

    double plan1ContainerCpuUtilityStdev = plan1.getContainerCpuUtilityStdev();
    double plan2ContainerCpuUtilityStdev = plan2.getContainerCpuUtilityStdev();
    if (Double.compare(plan1ContainerCpuUtilityStdev, plan2ContainerCpuUtilityStdev) > 0) {
      plan1Points++;
    } else if (Double.compare(plan1ContainerCpuUtilityStdev, plan2ContainerCpuUtilityStdev) < 0) {
      plan2Points++;
    }

    // TODO: +log

    return plan1Points >= plan2Points ? plan1 : plan2;
  }

  /**
   * Get a new packing plan given an existing packing plan and component-level changes.
   * @return new packing plan
   */
  public PackingPlan repack(PackingPlan currentPackingPlan, Map<String, Integer> componentChanges) {
    PackingPlanBuilder ramFirstPlanBuilder = newPackingPlanBuilder(currentPackingPlan);
    PackingPlanBuilder cpuFirstPlanBuilder = newPackingPlanBuilder(currentPackingPlan);

    // Get the instances using FFD allocation
    try {
      ramFirstPlanBuilder = get2DVectorBinAllocation(ramFirstPlanBuilder, SortingStrategy.RAM_FIRST,
          currentPackingPlan, componentChanges);
      cpuFirstPlanBuilder = get2DVectorBinAllocation(cpuFirstPlanBuilder, SortingStrategy.CPU_FIRST,
          currentPackingPlan, componentChanges);

      PackingPlan ramFirstPlan = ramFirstPlanBuilder.build();
      PackingPlan cpuFirstPlan = cpuFirstPlanBuilder.build();

      return pickBetter(ramFirstPlan, cpuFirstPlan);
    } catch (ResourceExceededException e) {
      throw new PackingException("Could not repack instances into existing packing plan", e);
    }
  }

  @Override
  public PackingPlan repack(PackingPlan currentPackingPlan, int containers,
                            Map<String, Integer> componentChanges)
      throws PackingException, UnsupportedOperationException {
    throw new UnsupportedOperationException("TwoDimensionVectorBinPacking does not currently "
        + "support creating a new packing plan with a new number of containers.");
  }

  @Override
  public void close() {

  }

  /**
   * Sort the components in decreasing order based on their RAM requirements
   *
   * @return The sorted list of components and their RAM requirements
   */
  private List<ResourceRequirement> getSortedInstances(Set<String> componentNames,
                                                            SortingStrategy sortingStrategy) {
    List<ResourceRequirement> resourceRequirements = new ArrayList<>();
    Map<String, ByteAmount> ramMap = TopologyUtils.getComponentRamMapConfig(topology);
    Map<String, Double> cpuMap = TopologyUtils.getComponentCpuMapConfig(topology);

    for (String componentName : componentNames) {
//      Resource requiredResource = PackingUtils.getResourceRequirement(
//          componentName, ramMap, this.defaultInstanceResources,
//          this.maxContainerResources, 10);
      Resource requiredResource = null;
      resourceRequirements.add(new ResourceRequirement(componentName,
          requiredResource.getRam(), requiredResource.getCpu(), sortingStrategy));
    }
    Collections.sort(resourceRequirements, Collections.reverseOrder());

    return resourceRequirements;
  }

  /**
   * Get the instances' allocation based on the First Fit Decreasing algorithm
   *
   * @return Map &lt; containerId, list of InstanceId belonging to this container &gt;
   */
  private PackingPlanBuilder get2DVectorBinAllocation(PackingPlanBuilder planBuilder,
                                                      SortingStrategy sortingStrategy)
      throws ResourceExceededException {
    Map<String, Integer> parallelismMap = TopologyUtils.getComponentParallelism(topology);
    assignInstancesToContainers(planBuilder, parallelismMap, sortingStrategy);
    return planBuilder;
  }

  /**
   * Get the instances' allocation based on the First Fit Decreasing algorithm
   *
   * @return Map &lt; containerId, list of InstanceId belonging to this container &gt;
   */
  private PackingPlanBuilder get2DVectorBinAllocation(PackingPlanBuilder packingPlanBuilder,
                                                      SortingStrategy sortingStrategy,
                                                      PackingPlan currentPackingPlan,
                                                      Map<String, Integer> componentChanges)
      throws ResourceExceededException {
    this.numContainers = currentPackingPlan.getContainers().size();

    Map<String, Integer> componentsToScaleDown =
        PackingUtils.getComponentsToScale(componentChanges, PackingUtils.ScalingDirection.DOWN);
    Map<String, Integer> componentsToScaleUp =
        PackingUtils.getComponentsToScale(componentChanges, PackingUtils.ScalingDirection.UP);

    if (!componentsToScaleDown.isEmpty()) {
      removeInstancesFromContainers(packingPlanBuilder, componentsToScaleDown, sortingStrategy);
    }

    if (!componentsToScaleUp.isEmpty()) {
      assignInstancesToContainers(packingPlanBuilder, componentsToScaleUp, sortingStrategy);
    }

    return packingPlanBuilder;
  }

  /**
   * Assigns instances to containers
   *
   * @param planBuilder existing packing plan
   * @param parallelismMap component parallelism
   */
  private void assignInstancesToContainers(PackingPlanBuilder planBuilder,
                                           Map<String, Integer> parallelismMap,
                                           SortingStrategy sortingStrategy)
      throws ResourceExceededException {
    List<ResourceRequirement> resourceRequirements
        = getSortedInstances(parallelismMap.keySet(), sortingStrategy);
    for (ResourceRequirement resourceRequirement : resourceRequirements) {
      String componentName = resourceRequirement.getComponentName();
      int numInstance = parallelismMap.get(componentName);
      for (int j = 0; j < numInstance; j++) {
        place2DVectorBinInstance(planBuilder, componentName);
      }
    }
  }

  /**
   * Removes instances from containers during scaling down
   *
   * @param packingPlanBuilder existing packing plan
   * @param componentsToScaleDown scale down factor for the components.
   */
  private void removeInstancesFromContainers(PackingPlanBuilder packingPlanBuilder,
                                             Map<String, Integer> componentsToScaleDown,
                                             SortingStrategy sortingStrategy) {

    List<ResourceRequirement> resourceRequirements =
        getSortedInstances(componentsToScaleDown.keySet(), sortingStrategy);

    InstanceCountScorer instanceCountScorer = new InstanceCountScorer();
    ContainerIdScorer containerIdScorer = new ContainerIdScorer(false);

    for (ResourceRequirement resourceRequirement : resourceRequirements) {
      String componentName = resourceRequirement.getComponentName();
      int numInstancesToRemove = -componentsToScaleDown.get(componentName);
      List<Scorer<Container>> scorers = new ArrayList<>();

      scorers.add(new HomogeneityScorer(componentName, true));  // all-same-component containers
      scorers.add(instanceCountScorer);                         // then fewest instances
      scorers.add(new HomogeneityScorer(componentName, false)); // then most homogeneous
      scorers.add(containerIdScorer);                           // then highest container id

      for (int j = 0; j < numInstancesToRemove; j++) {
        packingPlanBuilder.removeInstance(scorers, componentName);
      }
    }
  }

  /**
   * Assign a particular instance to an existing container or to a new container
   *
   */
  private void place2DVectorBinInstance(PackingPlanBuilder planBuilder,
                                        String componentName) throws ResourceExceededException {
    if (this.numContainers == 0) {
      planBuilder.updateNumContainers(++numContainers);
    }

    try {
      planBuilder.addInstance(new ContainerIdScorer(), componentName);
    } catch (ResourceExceededException e) {
      planBuilder.updateNumContainers(++numContainers);
//      planBuilder.addInstance(numContainers, componentName);
    }
  }
}
