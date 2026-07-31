from forgeos.batch_projection import BatchProjection


def test_batch_projection_uses_router_costs_for_each_task_type():
    result = BatchProjection().project([
        {"name": "format", "type": "format"},
        {"name": "review", "type": "review"},
    ])

    assert result["total_tasks"] == 2
    assert result["cost_breakdown"][0]["estimated_cost"] == 0.0001
    assert result["cost_breakdown"][1]["estimated_cost"] == 0.03
    assert result["cheapest_route"] == "deferred"
