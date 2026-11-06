import itertools
import math
import unittest
from numerics import (block_scaled_gemm, expanded_ranges, moe_example, scan_filter_topk,
                      sinkhorn, physical_indices, decode_fp4_byte, attention_with_sink,
                      speculative_distribution, greedy_prefix_plan, teaching_cache_key,
                      chain_visibility_example)


class DeepSeekMathTests(unittest.TestCase):
    def test_block_scaling_matches_dequantized_matmul(self):
        a,b=[[1,2,3,4],[2,0,1,3]],[[2,1,0,1],[1,3,2,0]]
        sa,sb=[[0.5,1],[2,0.25]],[1,2]
        full_a=[[x*sa[i][k//2] for k,x in enumerate(row)] for i,row in enumerate(a)]
        full_b=[[x*sb[k//2] for k,x in enumerate(row)] for row in b]
        oracle=[[sum(x*y for x,y in zip(ar,br)) for br in full_b] for ar in full_a]
        got=block_scaled_gemm(a,b,sa,sb,2)
        self.assertEqual(got,oracle)
        self.assertEqual(got,[[10,15.5],[9.5,5]])

    def test_expanded_alignment_counts(self):
        self.assertEqual([b-a for a,b in expanded_ranges([3,9,12,14],4)],[3,5,0,2])
        with self.assertRaises(ValueError): expanded_ranges([3,2],4)

    def test_moe_routes_and_gradient(self):
        r=moe_example()
        for actual,expected in zip(r['output'],[5,7.2]):self.assertAlmostEqual(actual,expected)
        for actual,expected in zip(r['input_gradient'],[2.5,2.4]):self.assertAlmostEqual(actual,expected)
        self.assertEqual(len(r['expanded_rows']),4)

    def test_topk_all_block_orders(self):
        x=[3,1,8,2,7,5,6,4,9,0,12,10]
        for order in itertools.permutations(range(3)):
            self.assertEqual(scan_filter_topk(x,3,4,3,order)[0],[(12,10),(10,11),(9,8)])
        self.assertEqual(scan_filter_topk(x,3,4,3,[1,0,2])[1],17)
        self.assertEqual(scan_filter_topk(x,3,4,3,[2,0,1])[1],11)

    def test_topk_rejects_nonfinite_input(self):
        with self.assertRaises(ValueError): scan_filter_topk([1,float('nan')],1,2,1,[0])

    def test_sinkhorn_stochastic_and_bounded(self):
        p=sinkhorn([[1,2,3],[4,2,1],[2,1,5]])
        for row in p:self.assertAlmostEqual(sum(row),1,places=12)
        for j in range(3):self.assertAlmostEqual(sum(row[j] for row in p),1,places=12)
        vector=[-3,1,2]
        out=[sum(a*b for a,b in zip(row,vector)) for row in p]
        self.assertLessEqual(max(map(abs,out)),3)

    def test_page_indices(self):
        self.assertEqual(physical_indices([1,4,5],[2,5],4),[9,20,21])

    def test_fp4_nibbles_and_sign(self):
        self.assertEqual(decode_fp4_byte(0x62,0.5),[0.5,2])
        self.assertEqual(decode_fp4_byte(0xF9,1),[-0.5,-6])
        self.assertEqual(math.copysign(1,decode_fp4_byte(0x08,1)[0]),-1)

    def test_sink_does_not_change_reported_lse(self):
        plain=attention_with_sink([0,math.log(3)],[2,6])
        sink=attention_with_sink([0,math.log(3)],[2,6],math.log(2))
        self.assertAlmostEqual(plain['output'],5)
        self.assertAlmostEqual(sink['output'],10/3)
        self.assertAlmostEqual(plain['lse_without_sink'],sink['lse_without_sink'])
        self.assertEqual(attention_with_sink([],[])['lse_without_sink'],-math.inf)

    def test_speculative_probability_identity(self):
        for p,q in [([.6,.4],[.5,.5]),([.2,.3,.5],[.4,.4,.2]),([1,0],[.5,.5]),([.5,.5],[.5,.5])]:
            got=speculative_distribution(p,q)
            for a,b in zip(got['output_distribution'],p):self.assertAlmostEqual(a,b)

    def test_retroactive_selection_is_biased(self):
        wrong_a=.5*1+.5*.6
        self.assertAlmostEqual(wrong_a,.8)
        self.assertNotAlmostEqual(wrong_a,.6)

    def test_hardware_aware_prefix_budget(self):
        plan=greedy_prefix_plan([[.9,.6,.4],[.7,.7,.2]],{2:100,3:90,4:80,5:60,6:55,7:48,8:40})
        self.assertEqual(plan['lengths'],[1,1])
        self.assertAlmostEqual(plan['trace'][-1]['throughput'],288)

    def test_cache_inputs_have_semantic_order(self):
        a=teaching_cache_key('x',['-DA=1','-DA=2'],'v',{},'1')
        b=teaching_cache_key('x',['-DA=2','-DA=1'],'v',{},'1')
        c=teaching_cache_key('x',['-DA=1','-DA=2'],'v',{},'2')
        self.assertNotEqual(a,b);self.assertNotEqual(a,c)

    def test_pending_chunk_read_waits(self):
        r=chain_visibility_example()
        self.assertEqual([x['status'] for x in r['tail_committed']],['retry','retry','ok'])
        self.assertTrue(all(x['value']==(1,'new') for x in r['all_acked']))


if __name__ == '__main__':unittest.main()
