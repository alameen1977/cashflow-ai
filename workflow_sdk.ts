import { workflow, node, trigger, sticky, ifElse, languageModel, outputParser, expr } from '@n8n/workflow-sdk';

const dailyTrigger = trigger({
  type: 'n8n-nodes-base.scheduleTrigger',
  version: 1.4,
  config: {
    name: 'Daily 08:45 Trigger',
    parameters: { rule: { interval: [{ field: 'days', daysInterval: 1, triggerAtHour: 8, triggerAtMinute: 45 }] } },
    position: [0, 200]
  },
  output: [{}]
});

const manualRun = trigger({
  type: 'n8n-nodes-base.manualTrigger',
  version: 1,
  config: { name: 'Run Demo Now', position: [0, 400] },
  output: [{}]
});

const runSettings = node({
  type: 'n8n-nodes-base.set',
  version: 3.5,
  config: {
    name: 'Run Settings',
    parameters: {
      mode: 'manual',
      assignments: {
        assignments: [
          { id: 'api', name: 'api_base', value: 'https://cashflow-ai-api-bf23.onrender.com', type: 'string' },
          { id: 'asof', name: 'as_of', value: '2013-09-15', type: 'string' },
          { id: 'tier', name: 'min_tier', value: 'MEDIUM', type: 'string' },
          { id: 'limit', name: 'max_invoices', value: 30, type: 'number' }
        ]
      }
    },
    position: [240, 300]
  },
  output: [{ api_base: 'https://cashflow-ai-api-bf23.onrender.com', as_of: '2013-09-15', min_tier: 'MEDIUM', max_invoices: 30 }]
});

const getInvoices = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.5,
  config: {
    name: 'Get Risk-Ranked Open Invoices',
    retryOnFail: true,
    maxTries: 3,
    waitBetweenTries: 5000,
    parameters: {
      method: 'GET',
      url: expr('{{ $json.api_base }}/portfolio/open-invoices'),
      sendQuery: true,
      specifyQuery: 'keypair',
      queryParameters: {
        parameters: [
          { name: 'as_of', value: expr('{{ $json.as_of }}') },
          { name: 'min_tier', value: expr('{{ $json.min_tier }}') },
          { name: 'limit', value: expr('{{ $json.max_invoices }}') }
        ]
      },
      options: { timeout: 120000 }
    },
    position: [480, 300]
  },
  output: [{
    as_of: '2013-09-15', open_invoices_total: 90, total_open_amount: 5331.48, total_cash_at_risk: 2080.78, high_value_threshold: 85.94,
    invoices: [{ priority: 1, invoice_number: '2666514859', customer_id: '9181-HEKGV', customer_email: 'ap+9181-hekgv@example.com', amount: 99.82, invoice_date: '2013-08-27', due_date: '2013-09-26', days_to_due: 11, disputed: true, probability_late: 0.982, challenger_probability_late: 0.993, expected_days_late_if_late: 21.0, cash_at_risk: 98.04, risk_tier: 'HIGH', recommended_action: 'Firm reminder now + phone call from account manager', needs_human_approval: true, reasons: ['invoice is under dispute', 'often paid late in the past'] }]
  }]
});

const splitInvoices = node({
  type: 'n8n-nodes-base.splitOut',
  version: 1.1,
  config: {
    name: 'One Item per Invoice',
    parameters: { fieldToSplitOut: 'invoices', include: 'noOtherFields' },
    position: [720, 200]
  },
  output: [{ as_of: '2013-09-15', priority: 1, invoice_number: '2666514859', customer_id: '9181-HEKGV', customer_email: 'ap+9181-hekgv@example.com', amount: 99.82, due_date: '2013-09-26', days_to_due: 11, disputed: true, probability_late: 0.982, challenger_probability_late: 0.993, cash_at_risk: 98.04, risk_tier: 'HIGH', recommended_action: 'Firm reminder now + phone call from account manager', needs_human_approval: true, reasons: ['invoice is under dispute', 'often paid late in the past'] }]
});

const chatModel = languageModel({
  type: '@n8n/n8n-nodes-langchain.lmChatOpenAi',
  version: 1.3,
  config: {
    name: 'OpenAI Chat Model',
    parameters: { model: { __rl: true, mode: 'list', value: 'gpt-5-mini' } },
    position: [960, 420]
  }
});

const emailParser = outputParser({
  type: '@n8n/n8n-nodes-langchain.outputParserStructured',
  version: 1.3,
  config: {
    name: 'Email Format',
    parameters: { schemaType: 'fromJson', jsonSchemaExample: '{ "subject": "Friendly reminder: invoice 123 due 26 Sep", "body": "Dear customer, ..." }' },
    position: [1120, 420]
  }
});

const draftEmail = node({
  type: '@n8n/n8n-nodes-langchain.chainLlm',
  version: 1.9,
  config: {
    name: 'Draft Reminder Email (AI)',
    parameters: {
      promptType: 'define',
      hasOutputParser: true,
      text: expr(
        'Write a short payment reminder email (max 120 words) from the Accounts Receivable team of "CashFlow AI Demo Co." to a customer.\n' +
        'Invoice number: {{ $json.invoice_number }}\n' +
        'Amount: AED {{ $json.amount }}\n' +
        'Due date: {{ $json.due_date }} (days until due: {{ $json.days_to_due }}; negative means overdue)\n' +
        'Risk tier: {{ $json.risk_tier }}\n' +
        'Invoice disputed: {{ $json.disputed }}\n' +
        'Tone rules:\n' +
        '- MEDIUM risk: warm, friendly reminder before the due date.\n' +
        '- HIGH risk: polite but firm; ask the customer to confirm the payment date by reply, and say their account manager will call.\n' +
        '- If the invoice is disputed: do NOT demand payment; acknowledge the query and offer a call to resolve it quickly.\n' +
        '- Never threaten, never mention AI, risk scores or internal predictions.\n' +
        'Sign off as "Accounts Receivable Team, CashFlow AI Demo Co."'
      ),
      batching: { batchSize: 5, delayBetweenBatches: 500 }
    },
    subnodes: { model: chatModel, outputParser: emailParser },
    position: [960, 200]
  },
  output: [{ output: { subject: 'Invoice 2666514859 - let us resolve your query', body: 'Dear customer, ...' } }]
});

const buildAction = node({
  type: 'n8n-nodes-base.set',
  version: 3.5,
  config: {
    name: 'Build Action Record',
    parameters: {
      mode: 'manual',
      assignments: {
        assignments: [
          { id: 'a1', name: 'run_at', value: expr('{{ $now.toISO() }}'), type: 'string' },
          { id: 'a2', name: 'as_of', value: expr("{{ $('Run Settings').first().json.as_of }}"), type: 'string' },
          { id: 'a3', name: 'priority', value: expr("{{ $('One Item per Invoice').item.json.priority }}"), type: 'number' },
          { id: 'a4', name: 'invoice_number', value: expr("{{ $('One Item per Invoice').item.json.invoice_number }}"), type: 'string' },
          { id: 'a5', name: 'customer_id', value: expr("{{ $('One Item per Invoice').item.json.customer_id }}"), type: 'string' },
          { id: 'a6', name: 'customer_email', value: expr("{{ $('One Item per Invoice').item.json.customer_email }}"), type: 'string' },
          { id: 'a7', name: 'amount', value: expr("{{ $('One Item per Invoice').item.json.amount }}"), type: 'number' },
          { id: 'a8', name: 'due_date', value: expr("{{ $('One Item per Invoice').item.json.due_date }}"), type: 'string' },
          { id: 'a9', name: 'days_to_due', value: expr("{{ $('One Item per Invoice').item.json.days_to_due }}"), type: 'number' },
          { id: 'a10', name: 'probability_late', value: expr("{{ $('One Item per Invoice').item.json.probability_late }}"), type: 'number' },
          { id: 'a11', name: 'challenger_probability_late', value: expr("{{ $('One Item per Invoice').item.json.challenger_probability_late }}"), type: 'number' },
          { id: 'a12', name: 'cash_at_risk', value: expr("{{ $('One Item per Invoice').item.json.cash_at_risk }}"), type: 'number' },
          { id: 'a13', name: 'risk_tier', value: expr("{{ $('One Item per Invoice').item.json.risk_tier }}"), type: 'string' },
          { id: 'a14', name: 'reasons', value: expr("{{ $('One Item per Invoice').item.json.reasons.join('; ') }}"), type: 'string' },
          { id: 'a15', name: 'recommended_action', value: expr("{{ $('One Item per Invoice').item.json.recommended_action }}"), type: 'string' },
          { id: 'a16', name: 'needs_approval', value: expr("{{ $('One Item per Invoice').item.json.needs_human_approval }}"), type: 'boolean' },
          { id: 'a17', name: 'email_subject', value: expr('{{ $json.output.subject }}'), type: 'string' },
          { id: 'a18', name: 'email_body', value: expr('{{ $json.output.body }}'), type: 'string' }
        ]
      }
    },
    position: [1280, 200]
  },
  output: [{ run_at: '2026-10-14T08:45:00.000+04:00', as_of: '2013-09-15', priority: 1, invoice_number: '2666514859', customer_id: '9181-HEKGV', customer_email: 'ap+9181-hekgv@example.com', amount: 99.82, due_date: '2013-09-26', days_to_due: 11, probability_late: 0.982, challenger_probability_late: 0.993, cash_at_risk: 98.04, risk_tier: 'HIGH', reasons: 'invoice is under dispute; often paid late in the past', recommended_action: 'Firm reminder now + phone call from account manager', needs_approval: true, email_subject: 'Invoice 2666514859', email_body: 'Dear customer, ...' }]
});

const approvalCheck = ifElse({
  version: 2.3,
  config: {
    name: 'Needs Human Approval?',
    parameters: {
      conditions: {
        combinator: 'and',
        options: { caseSensitive: true, leftValue: '', typeValidation: 'strict', version: 2 },
        conditions: [{ leftValue: expr('{{ $json.needs_approval }}'), rightValue: '', operator: { type: 'boolean', operation: 'true', singleValue: true } }]
      }
    },
    position: [1520, 200]
  }
});

const markPending = node({
  type: 'n8n-nodes-base.set',
  version: 3.5,
  config: {
    name: 'Hold for Manager Approval',
    parameters: {
      mode: 'manual',
      includeOtherFields: true,
      assignments: { assignments: [{ id: 's1', name: 'status', value: 'PENDING_APPROVAL', type: 'string' }] }
    },
    position: [1760, 100]
  },
  output: [{ status: 'PENDING_APPROVAL' }]
});

const markReady = node({
  type: 'n8n-nodes-base.set',
  version: 3.5,
  config: {
    name: 'Mark Ready to Send',
    parameters: {
      mode: 'manual',
      includeOtherFields: true,
      assignments: { assignments: [{ id: 's2', name: 'status', value: 'READY_TO_SEND', type: 'string' }] }
    },
    position: [1760, 300]
  },
  output: [{ status: 'READY_TO_SEND' }]
});

const saveAction = node({
  type: 'n8n-nodes-base.dataTable',
  version: 1.1,
  config: {
    name: 'Save Collection Action',
    parameters: {
      resource: 'row',
      operation: 'insert',
      dataTableId: { __rl: true, mode: 'id', value: 'qwTY4jLyao1f2sxF', cachedResultName: 'collections_actions' },
      columns: {
        mappingMode: 'autoMapInputData',
        value: {},
        schema: [
          { id: 'run_at', displayName: 'run_at', required: false, defaultMatch: false, display: true, type: 'dateTime', canBeUsedToMatch: true },
          { id: 'as_of', displayName: 'as_of', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'priority', displayName: 'priority', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'invoice_number', displayName: 'invoice_number', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'customer_id', displayName: 'customer_id', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'customer_email', displayName: 'customer_email', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'amount', displayName: 'amount', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'due_date', displayName: 'due_date', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'days_to_due', displayName: 'days_to_due', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'probability_late', displayName: 'probability_late', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'challenger_probability_late', displayName: 'challenger_probability_late', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'cash_at_risk', displayName: 'cash_at_risk', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'risk_tier', displayName: 'risk_tier', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'reasons', displayName: 'reasons', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'recommended_action', displayName: 'recommended_action', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'needs_approval', displayName: 'needs_approval', required: false, defaultMatch: false, display: true, type: 'boolean', canBeUsedToMatch: true },
          { id: 'status', displayName: 'status', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'email_subject', displayName: 'email_subject', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'email_body', displayName: 'email_body', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true }
        ]
      }
    },
    position: [2000, 200]
  },
  output: [{ id: 1, createdAt: '2026-10-14T08:45:00.000Z', updatedAt: '2026-10-14T08:45:00.000Z' }]
});

const getForecast = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.5,
  config: {
    name: 'Get Cash Forecast',
    executeOnce: true,
    retryOnFail: true,
    maxTries: 3,
    waitBetweenTries: 5000,
    parameters: {
      method: 'GET',
      url: expr("{{ $('Run Settings').item.json.api_base }}/portfolio/cash-forecast"),
      sendQuery: true,
      specifyQuery: 'keypair',
      queryParameters: {
        parameters: [
          { name: 'as_of', value: expr("{{ $('Run Settings').item.json.as_of }}") },
          { name: 'weeks', value: '4' }
        ]
      },
      options: { timeout: 120000 }
    },
    position: [720, 460]
  },
  output: [{ as_of: '2013-09-15', open_amount: 5331.48, weeks: [{ week_start: '2013-09-15', forecast_ai: 1445.26, forecast_due_date_rule: 1260.64, actual_collected: 1651.03 }] }]
});

const buildSummary = node({
  type: 'n8n-nodes-base.set',
  version: 3.5,
  config: {
    name: 'Build Daily CFO Summary',
    parameters: {
      mode: 'manual',
      assignments: {
        assignments: [
          { id: 'd1', name: 'run_at', value: expr('{{ $now.toISO() }}'), type: 'string' },
          { id: 'd2', name: 'as_of', value: expr("{{ $('Get Risk-Ranked Open Invoices').item.json.as_of }}"), type: 'string' },
          { id: 'd3', name: 'open_invoices_total', value: expr("{{ $('Get Risk-Ranked Open Invoices').item.json.open_invoices_total }}"), type: 'number' },
          { id: 'd4', name: 'total_open_amount', value: expr("{{ $('Get Risk-Ranked Open Invoices').item.json.total_open_amount }}"), type: 'number' },
          { id: 'd5', name: 'total_cash_at_risk', value: expr("{{ $('Get Risk-Ranked Open Invoices').item.json.total_cash_at_risk }}"), type: 'number' },
          { id: 'd6', name: 'cash_at_risk_pct', value: expr("{{ Math.round(1000 * $('Get Risk-Ranked Open Invoices').item.json.total_cash_at_risk / $('Get Risk-Ranked Open Invoices').item.json.total_open_amount) / 10 }}"), type: 'number' },
          { id: 'd7', name: 'actions_created', value: expr("{{ $('Get Risk-Ranked Open Invoices').item.json.invoices.length }}"), type: 'number' },
          { id: 'd8', name: 'high_risk_count', value: expr("{{ $('Get Risk-Ranked Open Invoices').item.json.invoices.filter(i => i.risk_tier === 'HIGH').length }}"), type: 'number' },
          { id: 'd9', name: 'approvals_needed', value: expr("{{ $('Get Risk-Ranked Open Invoices').item.json.invoices.filter(i => i.needs_human_approval).length }}"), type: 'number' },
          { id: 'd10', name: 'forecast_next_4_weeks', value: expr('{{ Math.round($json.weeks.reduce((s, w) => s + w.forecast_ai, 0) * 100) / 100 }}'), type: 'number' }
        ]
      }
    },
    position: [960, 460]
  },
  output: [{ run_at: '2026-10-14T08:45:00.000+04:00', as_of: '2013-09-15', open_invoices_total: 90, total_open_amount: 5331.48, total_cash_at_risk: 2080.78, cash_at_risk_pct: 39, actions_created: 30, high_risk_count: 25, approvals_needed: 2, forecast_next_4_weeks: 4755.28 }]
});

const saveSummary = node({
  type: 'n8n-nodes-base.dataTable',
  version: 1.1,
  config: {
    name: 'Save Daily Summary',
    parameters: {
      resource: 'row',
      operation: 'insert',
      dataTableId: { __rl: true, mode: 'id', value: 'K6mJT0r9XihHZqdI', cachedResultName: 'daily_collection_runs' },
      columns: {
        mappingMode: 'autoMapInputData',
        value: {},
        schema: [
          { id: 'run_at', displayName: 'run_at', required: false, defaultMatch: false, display: true, type: 'dateTime', canBeUsedToMatch: true },
          { id: 'as_of', displayName: 'as_of', required: false, defaultMatch: false, display: true, type: 'string', canBeUsedToMatch: true },
          { id: 'open_invoices_total', displayName: 'open_invoices_total', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'total_open_amount', displayName: 'total_open_amount', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'total_cash_at_risk', displayName: 'total_cash_at_risk', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'cash_at_risk_pct', displayName: 'cash_at_risk_pct', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'actions_created', displayName: 'actions_created', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'high_risk_count', displayName: 'high_risk_count', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'approvals_needed', displayName: 'approvals_needed', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true },
          { id: 'forecast_next_4_weeks', displayName: 'forecast_next_4_weeks', required: false, defaultMatch: false, display: true, type: 'number', canBeUsedToMatch: true }
        ]
      }
    },
    position: [1200, 460]
  },
  output: [{ id: 1, createdAt: '2026-10-14T08:45:00.000Z', updatedAt: '2026-10-14T08:45:00.000Z' }]
});

const overviewNote = sticky(
  '## CashFlow AI - Daily Collections Autopilot\n' +
  'Every morning: pulls open invoices ranked by **cash at risk** from the ML prediction API, ' +
  'drafts a tone-appropriate reminder for each with AI, and routes high-value HIGH-risk accounts to a manager for approval before anything is sent.\n\n' +
  '**Demo mode:** `as_of` in *Run Settings* simulates "today" inside the 2013 dataset.\n\n' +
  'API + model code: github.com/alameen1977/cashflow-ai',
  [],
  { color: 4, position: [0, -160], width: 560, height: 260 }
);

const controlNote = sticky(
  '### Control step\nHIGH-risk invoices above the 90th-percentile amount are **held for manager approval** - the automation never sends a firm message to a key account without a human sign-off. Everything else is marked READY_TO_SEND.',
  [approvalCheck, markPending, markReady],
  { color: 3 }
);

export default workflow('cashflow-ai-collections', 'CashFlow AI - Daily Collections Autopilot')
  .add(dailyTrigger)
  .to(runSettings)
  .add(manualRun)
  .to(runSettings)
  .add(runSettings)
  .to(getInvoices)
  .add(getInvoices)
  .to(splitInvoices)
  .to(draftEmail)
  .to(buildAction)
  .to(approvalCheck
    .onTrue(markPending.to(saveAction))
    .onFalse(markReady.to(saveAction)))
  .add(getInvoices)
  .to(getForecast)
  .to(buildSummary)
  .to(saveSummary)
  .add(overviewNote)
  .add(controlNote)
  .group('Fetch risk scores', [runSettings, getInvoices], { description: 'Calls the CashFlow AI prediction API for open invoices ranked by cash at risk (probability late x amount).' })
  .group('Draft reminders with AI', [splitInvoices, draftEmail, chatModel, emailParser, buildAction], { description: 'One item per invoice; AI writes a reminder whose tone matches the risk tier and dispute status.' })
  .group('Daily CFO summary', [getForecast, buildSummary, saveSummary], { description: 'Logs open amount, cash at risk, approvals needed and the 4-week AI cash forecast.' });
